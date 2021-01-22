import numpy as np
from scipy.misc import imresize
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
import keras.backend as K
from keras.backend import tensorflow_backend
import tensorflow as tf

from model.SEResNeXt import SEResNeXt

import json
import configparser



def arr_resize(arr, size):
    '''
    Resize Colored image array.
    '''
    resized_arr = np.empty((arr.shape[0],size,size,3))
    for idx, elem in enumerate(arr):
        resized_arr[idx] = imresize(elem, (size,size,3), interp='bilinear')

    return resized_arr
   
  
## Load parameters
inifile = configparser.ConfigParser()
inifile.read("./config.ini")
size = int(inifile.get('cifar10','size'))
num_classes = int(inifile.get('cifar10','num_classes'))
batch_size = int(inifile.get('cifar10','batch_size'))

## Memory setting
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)

## Data preparation
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

x_train = arr_resize(x_train, size)
x_test = arr_resize(x_test, size)

datagen = ImageDataGenerator(
    rescale = 1/255.
    , shear_range = 0.1
    , zoom_range = 0.1
    , channel_shift_range=0.1
    , rotation_range=15
    , width_shift_range=0.2
    , height_shift_range=0.2
    , horizontal_flip=True)
datagen.fit(x_train)

valid_datagen = ImageDataGenerator(rescale = 1/255.)
valid_datagen.fit(x_test)


## Create and compile a model
model = SEResNeXt(size, num_classes).model
sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)

def lr_scheduler(epoch):
    if epoch % 30 == 0:
        K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * 0.1)
    return K.eval(model.optimizer.lr)
change_lr = LearningRateScheduler(lr_scheduler)

model.compile(
    optimizer=sgd
    , loss='categorical_crossentropy'
    , metrics=['accuracy'])


## Set callbacks
model_save_name = "./trained_model/SEResNeXt"
filepath = model_save_name + "-{epoch:02d}-{val_acc:.3f}.h5"

csv_logger = CSVLogger('./logs/training.log')
checkpoint = ModelCheckpoint(
    filepath
    , monitor='val_acc'
    , verbose=5
    , save_best_only=True
    , mode='max'
)


## Model training
with open("{0}.json".format(model_save_name), 'w') as f:
    json.dump(json.loads(model.to_json()), f) # model.to_json() is a STRING of json

model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size)
    , steps_per_epoch=len(x_train) // batch_size
    , epochs=100
    , validation_data = valid_datagen.flow(x_test, y_test)
    , validation_steps=len(x_test) // batch_size
    , callbacks=[change_lr, csv_logger, checkpoint])

model.save_weights('{0}.h5'.format(model_save_name))
