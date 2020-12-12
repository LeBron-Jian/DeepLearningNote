'''
this script is LeNet model for Keras

'''
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D


def LeNet(input_shape, classes):
    '''

    :param input_shape: default (28, 28, 1)
    :param classes: default 10 （because the mnist class label is 10)
    :return:
    '''
    model = Sequential()
    model.add(Conv2D(32, (5, 5), strides=(1, 1), input_shape=input_shape,
                     padding='valid', activation='relu', kernel_initializer='uniform',
                     name='convolution_C1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='Subsampling_S2'))
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid',
                     activation='relu', kernel_initializer='uniform',
                     name='convolution_C3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='Subsampling_S4'))
    model.add(Flatten(name='Fullconnection_C5'))
    model.add(Dense(100, activation='relu', name='Fullconnection_F6'))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def LeNet_model(input_shape, classes):
    '''

    :param input_shape: default (28, 28, 1)
    :param classes: default 10 （because the mnist class label is 10)
    :return:
    '''
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), strides=(1, 1), input_shape=input_shape,
               padding='valid', activation='relu', kernel_initializer='uniform',
               name='convolution_C1')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), name='Subsampling_S2')(x)
    x = Conv2D(64, (5, 5), strides=(1, 1), padding='valid',
               activation='relu', kernel_initializer='uniform',
               name='convolution_C3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Subsampling_S4')(x)
    x = Flatten(name='Fullconnection_C5')(x)
    x = Dense(100, activation='relu', name='Fullconnection_F6')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs, output=x)

    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    LeNet(input_shape=(28, 28, 1), classes=10)
    # LeNet_model(input_shape=(28, 28, 1), classes=10)
