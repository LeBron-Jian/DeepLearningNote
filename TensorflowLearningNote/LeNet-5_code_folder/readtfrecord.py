# _*_coding:utf-8_*_
import tensorflow as tf
 
 
def read_record(filename):
    '''
    读取TFRecord文件
    :return:
    '''
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer([filename])
    # 创建一个reader来读取TFRecord文件中Example
    reader = tf.TFRecordReader()
    # 从文件中读取一个Example
    _, serialized_example = reader.read(filename_queue)
 
    # 用FixedLenFeature 将读入的Example解析成tensor
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'labels': tf.FixedLenFeature([], tf.int64)
        }
    )
 
    # tf.decode_raw将字符串解析成图像对应的像素数组
    images = tf.decode_raw(features['image_raw'], tf.uint8)
    labels = tf.cast(features['labels'], tf.int32)
    pixels = tf.cast(features['pixels'], tf.int32)
 
    init_op = tf.global_variables_initializer()
 
    with tf.Session() as sess:
        sess.run(init_op)
        # 启动多线程处理输入数据
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
 
        # 每次运行读出一个Example,当所有样例读取完之后，再次样例中程序会重头 读取
        for i in range(10):
            # 在会话中会取出image 和label
            image, label = sess.run([images, labels])
        coord.request_stop()
        coord.join(threads)
        print("end code")
 
 
if __name__ == '__main__':
    train_filename = 'mnistrecord/trainmnist28.tfrecords'
    test_filename = 'mnistrecord/testmnist28.tfrecords'
    read_record(filename=train_filename)
