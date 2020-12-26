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
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
import matplotlib as mpl
mpl.use('Agg')

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

from mymodel import *
from mydensenet121 import *

from keras.datasets import cifar10
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)

def train_model_with_data(X_train, Y_train, X_test, Y_test, batch_size, epochs, input_shape, classes):
    #model = DenseNet(input_shape, classes)
    #model = DenseNet_model(input_shape, classes, bottleneck=True, reduction=0.5)
    # model = ResNet_model(input_shape, classes)
    #model = VGGNet_model(input_shape, classes)
    # model = ZFNet_model(input_shape, classes)
    model = Alex_model(input_shape, classes)
    #model = LeNet_model(input_shape, classes)

    # initiate  optimizer
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
        metrics=['accuracy'])
    history = model.fit(X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, Y_test),
        shuffle = True,
        callbacks=[lr_reducer, early_stopper])

    score = model.evaluate(X_test, Y_test)
    print('loss accuracy is %s'%score)

    return history


def train_model_with_generator(X_train, Y_train, X_test, Y_test, train_nums, valid_nums, epochs, input_shape, classes, target_size, batch_size, class_mode='categorical'):
    model = DenseNet(input_shape, classes)
    #model = DenseNet_model(input_shape, classes, bottleneck=True, reduction=0.5)
    #model = ResNet_model(input_shape, classes)
    model = VGGNet_model(input_shape, classes)
    # model = ZFNet_model(input_shape, classes)
    #model = Alex_model(input_shape, classes)
    #model = LeNet_model(input_shape, classes)
    # 优化器
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
        metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # 是一个0~180的度数，用来指定随机选择图片的角度
            width_shift_range=0.1,  # 水平方向的随机移动程度
            height_shift_range=0.1,  # 竖直方向的随机移动程度
            horizontal_flip=True,  # 随机的对图片进行水平翻转，此参数用于水平翻转不影响图片语义的时候
            vertical_flip=False  # randomly flip images

            )

    train_data = train_datagen.flow(
        X_train,Y_train,
        batch_size = batch_size,
        ) 


    history = model.fit_generator(train_data, 
        samples_per_epoch=train_nums//batch_size,
        # samples_per_epoch=X_train.shape[0] // batch_size,
        nb_epoch=epochs,
        validation_data=(X_test, Y_test),
        nb_val_samples=valid_nums//batch_size,
        # nb_val_samples=20,
        callbacks=[lr_reducer, early_stopper])

    return history


def plot_train_Loss_Acc(history, save_path=r'lenetloss.jpg'):
    plt.figure(12)
    plt.subplot(121)
    plt.plot(history.history['acc'], 'ro')
    plt.plot(history.history['val_acc'], 'b')
    plt.title('Training and Validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train acc', 'val acc'], loc='upper left')

    plt.subplot(122)
    plt.plot(history.history['loss'], 'ro')
    plt.plot(history.history['val_loss'], 'b')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train loss', 'val loss'], loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)


def main(data_augmentation=True):
    target_size = (32, 32)
    input_shape, classes = (32, 32, 3), 10
    batch_size = 32
    epochs = 1600

    # the data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, classes)
    Y_test = to_categorical(y_test, classes)

    X_train, X_test = X_train.astype('float32'), X_test.astype('float32')

    # # subtract mean and normalize
    # mean_image = np.mean(X_train, axis=0)
    # X_train -= mean_image
    # X_test -= mean_image
    X_train /= 255
    X_test /= 255

    if not data_augmentation:
        history = train_model_with_data(X_train, Y_train, X_test, Y_test, batch_size, epochs, input_shape, classes)
        plot_train_Loss_Acc(history)
    else:
        train_nums, valid_nums = 50000, 10000
        history = train_model_with_generator(X_train, Y_train, X_test, Y_test, train_nums, valid_nums, epochs, input_shape, classes, target_size, batch_size)
        plot_train_Loss_Acc(history)

if __name__ == '__main__':
    start_time = time.time()
    main(False)
    # DenseNet_model(input_shape=(227, 227, 3), classes=1000, bottleneck=True, reduction=0.5)
    # ResNet_model(input_shape=(224, 224, 3), classes=1000)
    # VGGNet_model(input_shape=(227, 227, 3), classes=1000)
    # ZFNet_model(input_shape=(227, 227, 3), classes=1000)
    # Alex_model(input_shape=(227, 227, 3), classes=1000)
    # LeNet_model(input_shape=(28, 28, 3), classes=1000)
    print('all time is %s'%(time.time() - start_time))
