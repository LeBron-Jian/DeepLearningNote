# 提取图片中的 bottleneck 特征
'''
步骤：1，载入图片
      2，灌入 pre_model 的权重
      3，得到 bottleneck  feature
'''
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
 
# 载入图片  图片生成器初始化
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.applications.vgg16 import VGG16
import keras
 
def save_bottleneck_features():
    model = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
    print('load model ok')
 
    datagen = ImageDataGenerator(rescale=1./255)
 
    # 训练集图像生成器
    train_generator = datagen.flow_from_directory(
        'data/mytrain',
        target_size=(150, 150),
        batch_size=16,
        class_mode=None,
        shuffle=False
    )
 
    # 验证集图像生成器
    test_generator = datagen.flow_from_directory(
        'data/mytest',
        target_size=(150, 150),
        batch_size=16,
        class_mode=None,
        shuffle=False
    )
    print('increase image ok')
 
    # 灌入 pre_model 的权重
    WEIGHTS_PATH = ''
    model.load_weights('data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    print('load pre model OK ')
 
    # 得到 bottleneck feature
    bottleneck_features_train = model.predict_generator(train_generator, 500)
    # 核心，steps是生成器要返回数据的轮数，每个epoch含有500张图片，与model.fit(samples_per_epoch)相对
    # 如果这样的话，总共有 500*32个训练样本
    np.save('bottleneck_features_train.npy', 'w', bottleneck_features_train)
 
    bottleneck_features_validation = model.predict_generator(test_generator, 100)
    # 与model.fit(nb_val_samples)相对，一个epoch有800张图片，验证集
    np.save('bottleneck_features_validation.npy', 'w', bottleneck_features_validation)
 
 
 
def train_fine_tune():
    trainfile = 'data/model/train.npy'
    testfile = 'data/model/validation.npy'
    # (1）  导入 bottleneck features数据
    train_data = np.load(trainfile)
    print(train_data.shape)  # (8000, 4, 4, 512)
    # train_data = train_data.reshape(train_data.shape[0], 150, 150, 3)
    # the features were saved in order, so recreating the labels is easy
    train_labels = np.array(
        [0]*100 + [1]*100 + [2]*100 + [3]*100 + [4]*100
    )
    validation_data = np.load(testfile)
    print(validation_data.shape)  # (1432, 4, 4, 512)
    validation_labels = np.array(
        [0]*20 + [1]*20 + [2]*20 + [3]*20 + [4]*20
    )
 
    # (2） 设置标签，并规范成Keras默认格式
    train_labels = keras.utils.to_categorical(train_labels, 5)
    validation_labels = keras.utils.to_categorical(validation_labels, 5)
    print(train_labels.shape, validation_labels.shape)  # (8000, 5) (1432, 5)
    # (3) 写“小网络”的网络结构
    model = Sequential()
    # train_data.shape[1:]
    model.add(Flatten(input_shape=(4, 4, 512)))  #4*4*512
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))  # 二分类
    model.add(Dense(5, activation='softmax'))  # 多分类
 
    # (4) 设置参数并训练
    model.compile(loss='categorical_crossentropy',  # 两分类是  binary_crossentropy
                  optimizer='rmsprop',
                  metrics=['accuracy'])
 
    model.fit(train_data, train_labels,
              nb_epoch=50, batch_size=16,
              validation_data=(validation_data, validation_labels))
    model.save_weights('bottleneck_fc_model.h5')
 
if __name__ == '__main__':
    # save_bottleneck_features()
    train_fine_tune()
    # print('over')
