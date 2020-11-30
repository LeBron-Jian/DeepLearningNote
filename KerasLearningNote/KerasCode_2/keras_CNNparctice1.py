#_*_coding:utf-8_*_
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.set_image_dim_ordering('th')
 
# 简单的三层卷积加上ReLU激活函数，再接一个max-pooling层
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
#the model so far outputs 3D feature maps (height, width, features)
# 然后我们接了两个全连接网络，并以单个神经元和Sigmoid激活结束模型
# 这种选择会产生一个二分类的结果，与这种配置项适应，损失函数选择binary_crossentropy
# this converts our 3D feature maps to 1D feature vectors
model.add(Flatten())
# 添加隐藏层神经元的数量和激活函数
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
 
# next read data
# 使用 .flow_from_directory() 来从我们的jpgs图片中直接产生数据和标签
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
# this is the augmentation configuration we will use for testing only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)
 
# this is a generator that will read pictures found in subfliders
# of 'data/train'. and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    'DogsVSCats/train',  # this is the target directory
    target_size=(150, 150),  # all image will be resize to 150*150
    batch_size=32,
    class_mode='binary'
)  # since we use binary_crossentropy loss,we need binary labels
 
# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    'DogsVSCats/valid',  # this is the target directory
    target_size=(150, 150),  # all image will be resize to 150*150
    batch_size=32,
    class_mode='binary'
)
 
# 然后我们可以用这个生成器来训练网络了。
model.fit_generator(
    train_generator,
    samples_per_epoch=2000,
    nb_epoch=50,
    validation_data=validation_generator,
    nb_val_samples=800
)
model.save_weights('first_try.h5') #always save your weights after training or duraing trianing
