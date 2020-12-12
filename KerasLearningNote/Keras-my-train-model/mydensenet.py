'''
    this script is DenseNet model for Keras
    link: https://www.jianshu.com/p/274d050d517e

'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.layers import Input, Activation
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D
from keras.regularizers import l2
from keras.layers import Concatenate, Add, add, concatenate
from keras.layers.normalization import BatchNormalization

import keras.backend as K


def conv_block(input_shape, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    '''
        Apply BatchNorm, Relu, 3*3 Conv2D, optional bottleneck block and dropout
        Args:
            input_shape: Input keras tensor
            nb_filter: number of filters
            bottleneck: add bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)

    '''
    # 表示特征轴，因为连接和BN都是对特征轴来说
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_shape)
    x = Activation('relu')(x)

    # bottleneck 表示是否使用瓶颈层，也就是使用1*1的卷积层将特征图的通道数进行压缩
    if bottleneck:
        inter_channel = nb_filter * 4
        # He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，其中fan_in权重张量的扇入
        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same',
            use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', 
        use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(input_shape, nb_filter, compression=1.0, weight_decay=1e-4, is_max=False):
    '''
        过渡层是用来连接两个 dense block,同时在最后一个dense block的尾部不需要使用过渡层
        按照论文的说法：过渡层由四部分组成：
            BatchNormalization ReLU 1*1Conv  2*2Maxpooling
        Apply BatchNorm, ReLU , Conv2d, optional compression, dropout and Maxpooling2D
        Args:
            input_shape: keras tensor
            nb_filter: number of filters
            compression: caculated as 1-reduction, reduces the number of features maps in the transition block
                    (compression_rate 表示压缩率，将通道数进行调整)
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        return :
            keras tensor, after applying batch_norm, relu-conv, dropout maxpool
    '''

    # 表示特征轴，因为连接和BN都是对特征轴来说
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_shape)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal',
        padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # 论文提出使用均值池化层来做下采样，不过在边缘提取方面，最大池化层效果应该更好，可以加上接口
    if is_max:
        x = Maxpooling2D((2, 2), strides=(2, 2))(x)
    else:

        x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def dense_block(input_shape, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None,
    weight_decay=1e-4, grow_nb_filters=True, return_concat_list=False):
    '''
        Bulid a dense_block where the output of each conv_block is fed to subsequent ones
        此处使用循环实现了Dense Block 的密集连接

        Args:
            input_shape: keras tensor
            nb_layers: the number of layers of conv_block to append to the model
            nb_filter: number of filters
            growth_rate: growth rate
            weight_decay: weight decay factor
            grow_nv_filters: flag to decode to allow number of filters to grow
            return_concat_list: return the list of feature maps along with the actual output

        returns:
            keras tensor with nv_layers of conv_block append

        其中 x=concatenate([x, cb], axis=concat_axis)操作使得x在每次循环中始终维护一个全局状态
        第一次循环输入为 x， 输出为 cb1，第二次输入为 cb=[x,cb1]，输出为cb2，第三次输入为cb=[x,cb1,cb2]，输出为cb3
        以此类推，增长率为growth_rate 其实就是每次卷积时使用的卷积核个数，也就是最后输出的通道数。
    '''
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1

    x_list = [input_shape]

    for i in range(nb_layers):
        cb = conv_block(input_shape, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)

        x = concatenate([input_shape, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def DenseNet_model(input_shape, classes, depth=40, nb_dense_block=3, growth_rate=12, include_top=True,
        nb_filter=-1, nb_layers_per_block=[6, 12, 32, 32], bottleneck=False, reduction=0.0, dropout_rate=None,
        weight_decay=1e-4, subsample_initial_block=False, activation='softmax'):
    '''
        Build the DenseNet model

        Args:
            classes: number of classes
            input_shape: tuple of shape (channels, rows, columns) or (rows, columns, channels)
            include_top: flag to include the final Dense layer
            depth: number or layers
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
            nb_layers_per_block: number of layers in each dense block.
                    Can be a -1, positive integer or a list.
                    If -1, calculates nb_layer_per_block from the depth of the network.
                    If positive integer, a set number of layers per dense block.
                    If list, nb_layer is used as provided. Note that list size must
                    be (nb_dense_block + 1)
            bottleneck: add bottleneck blocks
            reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
            dropout_rate: dropout rate
            weight_decay: weight decay rate
            subsample_initial_block: Set to True to subsample the initial convolution and
                    add a MaxPool2D before the dense blocks are added.
            subsample_initial:
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                    Note that if sigmoid is used, classes must be 1.
        Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1

    if type(nb_layers_per_block) is not list:
        print('nb_layers_per_block should be a list !!!')
        return 0

    final_nb_layer = nb_layers_per_block[-1]
    nb_layers = nb_layers_per_block[:-1]

    # compute initial nb_filter if -1 else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # initial convolution
    if subsample_initial_block:
        initial_kernel = (7, 7)
        initial_strides = (2, 2)
    else:
        initial_kernel = (3, 3)
        initial_strides = (1, 1)

    Inp = Input(shape=input_shape)
    x =Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
        strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(Inp)

    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = Maxpooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # add dense blocks
    for block_index in range(nb_dense_block-1):
        x, nb_filter = dense_block(x, nb_layers[block_index], nb_filter, growth_rate,
            bottleneck=bottleneck, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # add transition block
        x = transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # the last dense block does not have a transition_block
    x, nb_filter = dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
        dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dense(classes, activation=activation)(x)

    model = Model(Inp, output=x)
    model.summary()

    return model



if __name__ == '__main__':
    DenseNet_model(input_shape=(227, 227, 3), classes=1000, bottleneck=True,
        reduction=0.5)
