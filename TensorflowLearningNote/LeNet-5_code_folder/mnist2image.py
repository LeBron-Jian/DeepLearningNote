# _*_coding:utf-8_*_
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials import mnist
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import os
 
 
# save raw image
def save_raw():
    # read data from mnist. if data not exist, will download automatically
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # we save image raw data to data/raw/ folder
    # if the folder not there, create it
    save_dir = 'MNIST_data/raw/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
 
    # save 20 images in training dataset
    for i in range(20):
        # please attention，mnist.train.images[i, :] is ith image, sequence started from 0
        image_array = mnist.train.images[i, :]
        # the image in MNIST of TensorFlow, image is 784 length vector, we recover it to 28x28 image
        image_array = image_array.reshape(28, 28)
        # save image as mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
        filename = save_dir + 'mnist_train_%d.jpg' % i
        # save image_array as image
        # use Image.fromarray to convert image，then call save function to save
        # because Image.fromarray is not good to support float, we have to convert it to uint8 and then read as 'L'
        Image.fromarray((image_array * 255).astype('uint8'), mode='L').convert('RGB').save(filename)
 
 
def convert_mnist_img_raw(data, data1, save_path):
    for i in range(data.images.shape[0]):
        # TensorFlow中的MNIST图片是一个784维的向量，我们重新把它还原为28x28维的图像。
        image_array = data.images[i, :]
        image_array = image_array.reshape(28, 28)
        img_num = (image_array * 255).astype(np.uint8)
        # img_num = (image_array * 255).astype('uint8')
        label = data1.labels[i]
        # cv2.imshow('image', img)
        # cv2.waitKey(500)
        filename = save_path + '/{}_{}.jpg'.format(label, i)
        print(filename)
        cv2.imwrite(filename, img_num)
        # Image.fromarray(img_num, mode='L').save(filename)
 
 
def convert_mnist_img(data, data1, save_path):
    for i in range(data.images.shape[0]):
        # TensorFlow中的MNIST图片是一个784维的向量，我们重新把它还原为28x28维的图像。
        image_array = data.images[i, :]
        image_array = image_array.reshape(28, 28)
        img_num = (image_array * 255).astype(np.uint8)
        # img_num = (image_array * 255).astype('uint8')
        label = data1.labels[i]
        if not os.path.exists(os.path.join(save_path, str(label))):
            os.mkdir(os.path.join(save_path, str(label)))
        # cv2.imshow('image', img)
        # cv2.waitKey(500)
        filename = save_path + '/' + str(label) + '/{}_{}.jpg'.format(label, i)
        print(filename)
        cv2.imwrite(filename, img_num)
        # Image.fromarray(img_num, mode='L').save(filename)
 
 
 
 
if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist1 = input_data.read_data_sets('MNIST_data/')
    content_name = ['MNIST_train', 'MNIST_test', 'MNIST_validation']
    # print(mnist.validation.images.shape[0])
    for i in content_name:
        if not os.path.exists(i):
            os.mkdir(i)
    # convert_mnist_img_raw(mnist.validation, mnist1.validation, 'MNIST_validation')  # 55000
    convert_mnist_img(mnist.train, mnist1.train, 'MNIST_train')  # 55000
    print('convert training data to image complete')
    convert_mnist_img(mnist.test, mnist1.test, 'MNIST_test')  # 10000
    print('convert test data to image complete')
    convert_mnist_img(mnist.validation, mnist1.validation, 'MNIST_validation')  # 5000
    print('convert validation data to image complete')
