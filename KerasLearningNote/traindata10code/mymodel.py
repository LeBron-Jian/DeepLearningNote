# _*_coding:utf-8_*_
import time
import cv2
import os
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.layers import merge, Concatenate, concatenate, Add, add
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import keras.backend as K
from keras.regularizers import l2



def LeNet_model(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), input_shape=input_shape,
        padding='valid', activation='relu', kernel_initializer='uniform',
        name='C1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='S2'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),
        padding='valid', activation='relu', kernel_initializer='uniform',
        name='C3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='S4'))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', name='F5'))
    model.add(Dense(84, activation='relu', name='F6'))
    model.add(Dense(classes, activation='softmax', name='Pre'))

    model.summary()

    # write model image
    # plot_model(model, to_file='lenet.jpg', show_shapes=True, show_layer_names=False)
    return model


def Alex_model(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), input_shape=input_shape,
        padding='same', activation='relu', kernel_initializer='uniform',
        name='C1'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='M2'))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='C3'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='M4'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='C5'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='C6'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='C7'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='M8'))
    model.add(Flatten(name='F9'))
    model.add(Dense(4096, activation='relu', name='F10'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='F11'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax', name='Pre'))

    model.summary()

    return model


def ZFNet_model(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), input_shape=input_shape,
        padding='valid', activation='relu', kernel_initializer='uniform',
        name='C1'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='M2'))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='C3'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='M4'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='C5'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='C6'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='C7'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='M8'))
    model.add(Flatten(name='F9'))
    model.add(Dense(4096, activation='relu', name='F10'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='F11'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax', name='Pre'))

    model.summary()

    return model


def VGGNet_model(input_shape, classes):
    model = Sequential()

    # Block 1
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape,
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block1_conv1'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape,
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block1_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block2_conv1'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block2_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block3_conv1'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block3_conv2'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block3_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block4_conv1'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block4_conv2'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block4_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block5_conv1'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block5_conv2'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer='uniform',
        name='block5_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool'))
    
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='Fc1'))
    model.add(Dense(4096, activation='relu', name='Fc2'))
    model.add(Dense(classes, activation='softmax', name='Pre'))

    model.summary()

    return model


def identity_block(input_tensor, nb_filter, kernel_size=(3, 3)):
    '''
        直接相加，并不需要 1*1 卷积
        nb_filter：卷积核个数，需要按顺序指定3个，例如（64,64,256）
    '''
    nb_filter1, nb_filter2, nb_filter3 = nb_filter
    out = Conv2D(nb_filter1, kernel_size=(1, 1))(input_tensor)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(nb_filter2, kernel_size, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(nb_filter3, kernel_size=(1, 1))(out)
    out = BatchNormalization()(out)

    out = Add()([out, input_tensor])
    out = Activation('relu')(out)
    return out


def conv_block(input_tensor, nb_filter, kernel_size=(3, 3)):
    nb_filter1, nb_filter2, nb_filter3 = nb_filter

    out = Conv2D(nb_filter1, kernel_size=(1, 1))(input_tensor)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(nb_filter2, kernel_size, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(nb_filter3, kernel_size=(1, 1))(out)
    out = BatchNormalization()(out)

    x = Conv2D(nb_filter3, kernel_size=(1, 1))(input_tensor)
    x = BatchNormalization()(x)

    out = Add()([out, x])
    out = Activation('relu')(out)

    return out


def ResNet_model(input_shape, classes):
    inp = Input(input_shape)
    out = ZeroPadding2D((3, 3))(inp)

    # stage 1
    out = Conv2D(filters=64, kernel_size=(7, 7), subsample=(2, 2))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(out)

    # stage 2
    out = conv_block(out, [64, 64, 256])
    out = identity_block(out, [64, 64, 256])
    out = identity_block(out, [64, 64, 256])

    # stage 3
    out = conv_block(out, [128, 128, 512])
    out = identity_block(out, [128, 128, 512])
    out = identity_block(out, [128, 128, 512])
    out = identity_block(out, [128, 128, 512])

    # stage 4
    out = conv_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])

    # stage 5
    out = conv_block(out, [512, 512, 2048])
    out = identity_block(out, [512, 512, 2048])
    out = identity_block(out, [512, 512, 2048])

    out = AveragePooling2D(pool_size=(7, 7))(out)
    out = Flatten()(out)

    out = Dense(classes, activation='softmax')(out)

    model = Model(inp, out)
    model.summary()
    return model


def _conv_block(input_shape, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
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


def _transition_block(input_shape, nb_filter, compression=1.0, weight_decay=1e-4, is_max=False):
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


def _dense_block(input_shape, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None,
    weight_decay=1e-4, grow_nb_filters=True, return_concat_list=False):

    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1

    x_list = [input_shape]

    for i in range(nb_layers):
        cb = _conv_block(input_shape, growth_rate, bottleneck, dropout_rate, weight_decay)
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
        x, nb_filter = _dense_block(x, nb_layers[block_index], nb_filter, growth_rate,
            bottleneck=bottleneck, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # add transition block
        x = _transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # the last dense block does not have a transition_block
    x, nb_filter = _dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
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
    start_time = time.time()
    # DenseNet_model(input_shape=(227, 227, 3), classes=1000, bottleneck=True, reduction=0.5)
    # ResNet_model(input_shape=(224, 224, 3), classes=1000)
    # VGGNet_model(input_shape=(227, 227, 3), classes=1000)
    ZFNet_model(input_shape=(227, 227, 3), classes=10)
    #Alex_model(input_shape=(227, 227, 3), classes=10)
    # LeNet_model(input_shape=(28, 28, 3), classes=1000)
    print('all time is %s'%(time.time() - start_time))
