#_*_coding:utf-8_*_
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.applications.vgg16 import VGG16
import os
 
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
 
 
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
set_session(tf.Session(config=config))
 
 
model = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
print('load model ok')
 
datagen = ImageDataGenerator(rescale=1./255)
 
 
train_generator = datagen.flow_from_directory(
    '/data/lebron/data/mytrain',
    target_size=(150, 150),
    batch_size=4,
    class_mode=None,
    shuffle=False
)
 
 
test_generator = datagen.flow_from_directory(
    '/data/lebron/data/mytest',
    target_size=(150, 150),
    batch_size=4,
    class_mode=None,
    shuffle=False
)
print('increase image ok')
 
model.load_weights('/data/lebron/data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
print('load pre model OK ')
 
 
bottleneck_features_train = model.predict_generator(train_generator, 125)
 
np.save('/data/lebron/bottleneck_features_train.npy', bottleneck_features_train)
 
 
bottleneck_features_validation = model.predict_generator(test_generator, 25)
 
np.save('/data/lebron/bottleneck_features_validation.npy', bottleneck_features_validation)
 
print('game over')
