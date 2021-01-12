# _*_coding:utf-8_*_
import tensorflow as tf
import os
 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
 
# 1,输入数据的解析和预处理
def read_records(filename, resize_height, resize_width, type=None):
    '''
    解析record文件：源文件的图像数据是RGB，uint8 【0， 255】一般作为训练数据时，需要归一化到[0, 1]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param type: 选择图像数据的返回类型
                 None:默认将uint8-[0,255]转为float32-[0,255]
                 normalization:归一化float32-[0,1]
                 centralization:归一化float32-[0,1],再减均值中心化
    :return:
    '''
    # 创建文件队列，不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader 从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)  # 获得图像原始的数据
 
    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    tf_label = tf.cast(features['label'], tf.int32)
    # PS: 恢复原始图像数据，reshape的大小必须与保存之前的图像shape一致，否则出错
    tf_image = tf.reshape(tf_image, [resize_height, resize_width, 3])  # 设置图像的维度
 
    # 存储的图像类型为uint8，TensorFlow训练时数据必须是tf.float32
    if type is None:
        tf_image = tf.cast(tf_image, tf.float32)
    elif type == 'normalization':  # [1]若需要归一化请使用:
        # 仅当输入数据是uint8,才会归一化[0,255]
        # tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255.0)  # 归一化
    elif type == 'centralization':
        # 若需要归一化,且中心化,假设均值为0.5,请使用:
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5  # 中心化
 
        # 这里仅仅返回图像和标签
        # return tf_image, tf_height,tf_width,tf_depth,tf_label
    return tf_image, tf_label
 
 
def get_batch_images(images, labels, batch_size, labels_nums, one_hot=False,
                     shuffle=False, num_threads=1):
    '''
    :param images: 图像
    :param labels: 标签
    :param batch_size:
    :param labels_nums: 标签个数
    :param one_hot: 是否将labels转化为one_hot 的形式
    :param shuffle: 是否打乱顺序，一般train时，shuffle=True，验证时shuffle=False
    :param num_threads:
    :return: 返回batch的images和labels
    '''
    min_after_dequeue = 200
    # 保证 capacity必须大于 min_after_dequeue的参数值
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images, labels],
                                                            batch_size=batch_size,
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue,
                                                            num_threads=num_threads)
    else:
        images_batch, labels_batch = tf.train.batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    num_threads=num_threads)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch, labels_batch
