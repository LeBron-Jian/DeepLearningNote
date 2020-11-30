# 载入与模型网络构建
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
 
 
def built_model():
    # 载入与模型网络构建
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    # filter大小为3*3 数量为32个，原始图像大小3,150 150
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # this converts ours 3D feature maps to 1D feature vector
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))  # 几个分类就几个dense
    model.add(Activation('softmax'))  # 多分类
    # model.compile(loss='binary_corssentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
 
    # 优化器rmsprop：除学习率可调整外，建议保持优化器的其他默认参数不变
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
 
    model.summary()
    return model
 
 
def generate_data():
    '''
    flow_from_directory是计算数据的一些属性值，之后再训练阶段直接丢进去这些生成器。
    通过这个函数来准确数据，可以让我们的jpgs图片中直接产生数据和标签
    :return:
    '''
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        'data/mytrain',
        target_size=(150, 150),  # all images will be resized to 150*150
        batch_size=32,
        class_mode='categorical'  # 多分类
    )
 
    validation_generator = test_datagen.flow_from_directory(
        'data/mytest',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'  # 多分类
    )
    return train_generator, validation_generator
 
 
def train_model(model=None):
    if model is None:
        model = built_model()
        model.fit_generator(
            train_generator,
            # sampels_per_epoch 相当于每个epoch数据量峰值，
            # 每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
            samples_per_epoch=2000,
            nb_epoch=50,
            validation_data=validation_generator,
            nb_val_samples=800
        )
        model.save_weights('first_try_animal.h5')
 
 
if __name__ == '__main__':
    train_generator, validation_generator = generate_data()
    train_model()
    # 当loss出现负数，肯定是之前多分类的标签哪些设置的不对，
