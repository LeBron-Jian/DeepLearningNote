'''
this script is ZFNet model for Keras

'''
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam


def ZFNet(input_shape, classes):
    '''

    :param input_shape: default (227, 227, 3)
    :param classes: default 1000
    :return:
    '''
    model = Sequential()
    model.add(Conv2D(96, (7, 7), strides=(2, 2), input_shape=input_shape,
                     padding='valid', activation='relu', kernel_initializer='uniform',
                     name='convolution_C1'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Overlapping_M2'))
    model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same',
                     activation='relu', kernel_initializer='uniform',
                     name='convolution_C3'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Overlapping_M4'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform',
                     name='convolution_C5'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform',
                     name='convolution_C6'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform',
                     name='convolution_C7'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Overlapping_M8'))
    model.add(Flatten(name='Fullconnection_C9'))
    model.add(Dense(4096, activation='relu', name='Fullconnection_F10'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='Fullconnection_F11'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def ZFNet_model(input_shape, classes):
    '''

    :param input_shape: default (224, 224, 3)
    :param classes: default 1000
    :return:
    '''
    inputs = Input(shape=input_shape)
    x = Conv2D(96, (7, 7), strides=(2, 2), padding='valid', activation='relu',
               kernel_initializer='uniform', name='convolution_C1')(inputs)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Overlapping_M2')(x)

    x = Conv2D(256, (5, 5), strides=(2, 2), padding='same',
               activation='relu', kernel_initializer='uniform',
               name='convolution_C3')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Overlapping_M4')(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform',
               name='convolution_C5')(x)
    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform',
               name='convolution_C6')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform',
               name='convolution_C7')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Overlapping_M8')(x)

    x = Flatten(name='Fullconnection_C9')(x)
    x = Dense(4096, activation='relu', name='Fullconnection_F10')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='Fullconnection_F11')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs, output=x)

    # my_optimizer = Adam(0.001)
    # model.compile(optimizer='my_optimizer', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    # ZFNet(input_shape=(227, 227, 3), classes=1000)
    ZFNet_model(input_shape=(224, 224, 3), classes=1000)
