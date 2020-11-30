from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
 
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
 
 
def save_bottleneck_features():
    model = MobileNet(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
    print('load model ok')
    datagen = ImageDataGenerator(rescale=1. / 255)
 
    # train set image generator
    train_generator = datagen.flow_from_directory(
        '/data/lebron/data/mytrain',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
 
    # test set image generator
    test_generator = datagen.flow_from_directory(
        '/data/lebron/data/mytest',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
 
    # load weight
    model.load_weights(WEIGHTS_PATH_NO_TOP)
    print('load weight ok')
    # get bottleneck feature
    bottleneck_features_train = model.predict_generator(train_generator, 10)
    np.save(save_train_path, bottleneck_features_train)
 
    bottleneck_features_validation = model.predict_generator(test_generator, 2)
    np.save(save_test_path, bottleneck_features_validation)
     
 
def train_fine_tune():
    # load bottleneck features
    train_data = np.load(save_train_path)
    train_labels = np.array(
        [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100
    )
    validation_data = np.load(save_test_path)
    validation_labels = np.array(
        [0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20
    )
    # set labels
    train_labels = keras.utils.to_categorical(train_labels, 5)
    validation_labels = keras.utils.to_categorical(validation_labels, 5)
 
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
 
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
 
    model.fit(train_data, train_labels,
              nb_epoch=500, batch_size=25,
              validation_data=(validation_data, validation_labels))
 
 
if __name__ == '__main__':
    WEIGHTS_PATH = '/data/model/mobilenet_1_0_224_tf.h5'
    WEIGHTS_PATH_NO_TOP = '/data/model/mobilenet_1_0_224_tf_no_top.h5'
    save_train_path = '/data/bottleneck_features_train.npy'
    save_test_path = '/data/bottleneck_features_validation.npy'
    batch_size = 50
    save_bottleneck_features()
    train_data = np.load(save_train_path)
    validation_data = np.load(save_test_path)
    print(train_data.shape, validation_data.shape)
    train_fine_tune()
    print('game over')
