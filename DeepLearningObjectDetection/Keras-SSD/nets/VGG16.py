import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import merge, concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model

def VGG16(input_tensor):
    #----------------------------主干特征提取网络开始---------------------------#
    # SSD结构,net字典
    net = {} 
    # Block 1
    net['input'] = input_tensor
    # 300,300,3 -> 150,150,64
    net['conv1_1'] = Conv2D(64, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv1_1')(net['input'])
    net['conv1_2'] = Conv2D(64, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv1_2')(net['conv1_1'])
    net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool1')(net['conv1_2'])

    
    # Block 2
    # 150,150,64 -> 75,75,128
    net['conv2_1'] = Conv2D(128, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv2_1')(net['pool1'])
    net['conv2_2'] = Conv2D(128, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv2_2')(net['conv2_1'])
    net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool2')(net['conv2_2'])
    # Block 3
    # 75,75,128 -> 38,38,256
    net['conv3_1'] = Conv2D(256, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = Conv2D(256, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Conv2D(256, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool3')(net['conv3_3'])
    # Block 4
    # 38,38,256 -> 19,19,512
    net['conv4_1'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_3')(net['conv4_2'])
    net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool4')(net['conv4_3'])
    # Block 5
    # 19,19,512 -> 19,19,512
    net['conv5_1'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                                name='pool5')(net['conv5_3'])
    # FC6
    # 19,19,512 -> 19,19,1024
    net['fc6'] = Conv2D(1024, kernel_size=(3,3), dilation_rate=(6, 6),
                                     activation='relu', padding='same',
                                     name='fc6')(net['pool5'])

    # x = Dropout(0.5, name='drop6')(x)
    # FC7
    # 19,19,1024 -> 19,19,1024
    net['fc7'] = Conv2D(1024, kernel_size=(1,1), activation='relu',
                               padding='same', name='fc7')(net['fc6'])

    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    # 19,19,512 -> 10,10,512
    net['conv6_1'] = Conv2D(256, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='conv6_1')(net['fc7'])
    net['conv6_2'] = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(net['conv6_1'])
    net['conv6_2'] = Conv2D(512, kernel_size=(3,3), strides=(2, 2),
                                   activation='relu',
                                   name='conv6_2')(net['conv6_2'])

    # Block 7
    # 10,10,512 -> 5,5,256
    net['conv7_1'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same', 
                                   name='conv7_1')(net['conv6_2'])
    net['conv7_2'] = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(net['conv7_1'])
    net['conv7_2'] = Conv2D(256, kernel_size=(3,3), strides=(2, 2),
                                   activation='relu', padding='valid',
                                   name='conv7_2')(net['conv7_2'])
    # Block 8
    # 5,5,256 -> 3,3,256
    net['conv8_1'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='conv8_1')(net['conv7_2'])
    net['conv8_2'] = Conv2D(256, kernel_size=(3,3), strides=(1, 1),
                                   activation='relu', padding='valid',
                                   name='conv8_2')(net['conv8_1'])

    # Block 9
    # 3,3,256 -> 1,1,256
    net['conv9_1'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='conv9_1')(net['conv8_2'])
    net['conv9_2'] = Conv2D(256, kernel_size=(3,3), strides=(1, 1),
                                   activation='relu', padding='valid',
                                   name='conv9_2')(net['conv9_1'])
    #----------------------------主干特征提取网络结束---------------------------#
    return net