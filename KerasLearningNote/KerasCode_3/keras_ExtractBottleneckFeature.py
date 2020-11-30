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
model = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
print('load model ok')
 
datagen = ImageDataGenerator(rescale=1./255)
 
# 训练集图像生成器
train_generator = datagen.flow_from_directory(
    'data/mytrain',
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False
)
 
# 验证集图像生成器
test_generator = datagen.flow_from_directory(
    'data/mytest',
    target_size=(150, 150),
    batch_size=32,
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
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
 
bottleneck_features_validation = model.predict_generator(test_generator, 100)
# 与model.fit(nb_val_samples)相对，一个epoch有800张图片，验证集
np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
