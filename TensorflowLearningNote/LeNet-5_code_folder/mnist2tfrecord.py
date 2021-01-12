#_*_coding:utf-8_*_
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
 
# 生成整数的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
 
# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
def create_records(images, labels, num_example, outpout):
    '''
    实现将MNIST数据集转化为records
    注意：读取的图像数据默认为uint8，然后转化为tf的字符串型BytesList保存
    :return:
    '''
    # 训练图像的分辨率，作为example的属性
    pixels = images.shape[1]
    # 创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(outpout)
 
    # 将每张图片都转化为一个Example
    for i in range(num_example):
        # 将图像转化为字符串
        image_raw = images[i].tostring()
 
        # 将一个样例转化为Example，Protocal Buffer并将所有信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'pixels': _int64_feature(pixels),
                'labels': _int64_feature(np.argmax(labels[i])),
                'image_raw': _bytes_feature(image_raw)
            }
        ))
 
        # 将Example写入TFRecord文件
        writer.write(example.SerializeToString())
    print("data processing success")
    writer.close()
 
 
 
if __name__ == '__main__':
    if not os.path.exists('mnistrecord'):
        os.mkdir('mnistrecord')
    # 导入MNIST数据集
    mnist = input_data.read_data_sets('MNIST_data/', dtype=tf.uint8, one_hot=True)
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    train_num_example = mnist.train.num_examples
    # 存储train_TFRecord文件的地址
    train_filename = 'mnistrecord/trainmnist28.tfrecords'
    create_records(train_images, train_labels, train_num_example, train_filename)
 
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_num_example = mnist.test.num_examples
    # 存储train_TFRecord文件的地址
    test_filename = 'mnistrecord/testmnist28.tfrecords'
    create_records(test_images, test_labels, test_num_example, test_filename)
