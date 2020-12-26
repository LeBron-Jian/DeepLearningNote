#_*_coding:utf-8_*_
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Convolution2D
from keras.layers import AveragePooling2D, ZeroPadding2D
from keras.layers import add, Flatten, Activation, concatenate
from keras.optimizers import SGD
import numpy as np


def identity_block(input_tensor, nb_filter, kernel_size=3):
    '''
        直接相加，并不需要 1*1 卷积
        input_tensor：输入
        nb_filter：卷积核个数，需要按顺序指定3个，例如（64， 64， 256）
        kernel_size：卷积核大小
    '''
    nb_filter1, nb_filter2, nb_filter3 = nb_filter

    out = Conv2D(nb_filter1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(nb_filter2, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(nb_filter3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(out)
    out = BatchNormalization()(out)

    out = add([out, input_tensor])
    out = Activation('relu')(out)

    return out


def conv_block(input_tensor, nb_filter, kernel_size=3):
    nb_filter1, nb_filter2, nb_filter3 = nb_filter

    # 有人这里 strides=(2,2)
    out = Conv2D(nb_filter1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(nb_filter2, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(nb_filter3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(out)
    out = BatchNormalization()(out)

    shortcut = Conv2D(nb_filter3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    out = add([out, shortcut])
    out = Activation('relu')(out)

    return out


def resnet50(input_shape, classes):
    '''
        input_shape = (224, 224, 3)
    '''
    # define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    # stage 1
    X = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # stage 2
    X = conv_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # stage 3
    X = conv_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])


    # stage 4
    X = conv_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # stage 5
    X = conv_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    X = AveragePooling2D((7, 7))(X)
    # X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)
    X = Flatten()(X)
    X_out = Dense(classes, activation='softmax')(X)

    model = Model(X_input, X_out)

    model.summary()

    return model


resnet50(input_shape=(224, 224, 3), classes=1000)