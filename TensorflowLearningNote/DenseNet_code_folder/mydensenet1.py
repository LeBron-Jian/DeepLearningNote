'''
    the script is denseNet Model for keras
    link: https://blog.csdn.net/shi2xian2wei2/article/details/84425777
    
'''
import numpy as np
import keras
from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, Activation


def DenseLayer(x, nb_filter, bn_size=4, drop_rate=0.2):
    # Bottleneck layers
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(bn_size*nb_filter, (1, 1), strides=(1, 1), padding='same')(x)

    # composite function
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x)

    if drop_rate:
        x = Dropout(drop_rate)(x)

    return x


def TransitionLayer(x, compression=0.5, is_max=0):
    nb_filter = int(x.shape.as_list()[-1]*compression)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), padding='same')(x)
    if is_max:
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    else:
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    return x


def DenseBlock(x, nb_filter, growth_rate, drop_rate=0.2):
    for ii in range(nb_filter):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=3)

    return x

def DenseNet_model(input_shape, classes, growth_rate=12):
    inp = Input(shape=input_shape)
    x = Conv2D(growth_rate*2, (3, 3), strides=1, padding='same')(inp)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
    x = TransitionLayer(x)

    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
    x = TransitionLayer(x)

    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
    x = BatchNormalization(axis=3)(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(classes, activation='softmax')(x)

    model = Model(inp, x)
    model.summary()

    return model

if __name__ == '__main__':
    DenseNet_model(input_shape=(227, 227, 3), classes=1000)
