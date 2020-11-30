#_*_coding:utf-8_*_
'''
使用ImageDataGenerator 来生成图片，并将其保存在一个临时文件夹中
下面感受一下数据提升究竟做了什么事情。
'''
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
 
datagen = ImageDataGenerator(
    rotation_range=40,  # 是一个0~180的度数，用来指定随机选择图片的角度
    width_shift_range=0.2,  # 水平方向的随机移动程度
    height_shift_range=0.2,  # 竖直方向的随机移动程度
    rescale=1./255,  #将在执行其他处理前乘到整个图像上
    shear_range=0.2,  # 用来进行剪切变换的程度，参考剪切变换
    zoom_range=0.2,  # 用来进行随机的放大
    horizontal_flip=True,  # 随机的对图片进行水平翻转，此参数用于水平翻转不影响图片语义的时候
    fill_mode='nearest'  # 用来指定当需要进行像素填充，如旋转，水平和竖直位移的时候
)
 
pho_path = 'timg.jpg'
img = load_img(pho_path)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1, ) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
 
if not os.path.exists('preview'):
    os.mkdir('preview')
 
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='durant', save_format='jpg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefitely
