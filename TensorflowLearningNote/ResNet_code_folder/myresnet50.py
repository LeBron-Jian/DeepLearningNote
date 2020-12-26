'''
This script demonstrates how to build a deep residual network using the Keras functional APO

resnet_50 returns the deep residual network model(50 layers)

Please visit Kaiming He's github hompage:https://github.com/KaimingHe
for more information
'''
from keras.layers import Add, Concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input

import keras.backend as K


def identity_block(input_tensor, kernel_size, nb_filter, stage, block):
    '''
    the identity_block is the block that has no conv layer at shortcut
    直接相加，并不需要 1*1 卷积
    :param input_tensor:  输入
    :param nb_filter: 卷积核个数，需要按顺序指定3个，例如（64, 64, 256)
    :param kernel_size: 卷积核大小
    :return:
    '''
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        axis = 3
    else:
        axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    nb_filter1, nb_filter2, nb_filter3 = nb_filter

    out = Convolution2D(nb_filter1, 1, 1, dim_ordering=dim_ordering,
                        name=conv_name_base + '2a')(input_tensor)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                        dim_ordering=dim_ordering, name=conv_name_base + '2b')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter3, 1, 1, dim_ordering=dim_ordering,
                        name=conv_name_base + '2c')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2c')(out)

    # out = merge([out, input_tensor], mode='sum')
    out = Add()([out, input_tensor])
    # out = Concatenate()([out, input_tensor])
    out = Activation('relu')(out)
    return out


def conv_block(input_tensor, kernel_size, nb_filter, stage, block, strides=(2, 2)):
    '''
    conv_block is the block that has a conv layer at shortcut

    :param input_tensor: input tensor
    :param kernel_size: default 3, the kernel size of middle conv layer at main path
    :param nb_filter:  list of integers, the nb_filters of 3 conv layer at main path
    :param stage: integer, current stage label, used for generating layer names
    :param block: a, b,.... current block labels, used for generating layer names
    :param strides: default tuple (2, 2)
    Note that form stage 3: the first conv layer at main path is with subsample=(2,2)
    and the shortcut should has subsample=(2, 2) as well
    :return:
    '''
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        axis = 3
    else:
        axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    nb_filter1, nb_filter2, nb_filter3 = nb_filter

    out = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                        dim_ordering=dim_ordering, name=conv_name_base + '2a')(input_tensor)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                        dim_ordering=dim_ordering, name=conv_name_base + '2b')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter3, 1, 1, dim_ordering=dim_ordering,
                        name=conv_name_base + '2c')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2c')(out)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides, dim_ordering=dim_ordering,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=axis, name=bn_name_base + '1')(shortcut)

    # out = merge([out, shortcut], mode='sum')
    out = Add()([out, shortcut])
    # x = Add()([x, inputs])
    # out = Concatenate()([out, shortcut])
    out = Activation('relu')(out)
    return out


def resnet_50():
    '''
    this function returns the 50-layer residual network model
    you should load pretrained weights if you want to use it directly

    :return:
    '''
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        inp = Input(shape=(224, 224, 3))
        axis = 3
    else:
        inp = Input(shape=(3, 224, 224))
        axis = 1

    out = ZeroPadding2D((3, 3), dim_ordering=dim_ordering)(inp)

    # stage 1
    out = Convolution2D(64, 7, 7, subsample=(2, 2), dim_ordering=dim_ordering,
                        name='conv1')(out)
    out = BatchNormalization(axis=axis, name='bn_conv1')(out)
    out = Activation('relu')(out)
    out = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=dim_ordering)(out)

    # stage 2
    out = conv_block(out, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    out = identity_block(out, 3, [64, 64, 256], stage=2, block='b')
    out = identity_block(out, 3, [64, 64, 256], stage=2, block='c')

    # stage 3
    out = conv_block(out, 3, [128, 128, 512], stage=3, block='a')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='b')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='c')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='d')

    # stage 4
    out = conv_block(out, 3, [256, 256, 1024], stage=4, block='a')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='b')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='c')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='d')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='e')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='f')

    # stage 5
    out = conv_block(out, 3, [512, 512, 2048], stage=5, block='a')
    out = identity_block(out, 3, [512, 512, 2048], stage=5, block='b')
    out = identity_block(out, 3, [512, 512, 2048], stage=5, block='c')

    out = AveragePooling2D((7, 7), dim_ordering=dim_ordering)(out)
    out = Flatten()(out)
    out = Dense(1000, activation='softmax', name='fc1000')(out)

    model = Model(inp, out)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model



resnet_50()