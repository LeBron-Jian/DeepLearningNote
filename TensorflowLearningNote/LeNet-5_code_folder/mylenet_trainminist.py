# _*_coding:utf-8_*_
'''
LeNet是第一个成功应用于数字识别问题的卷积神经网络
LeNet模型总共有7层，在MNIST上LeNet-5模型可以达到99.2%的正确率
'''
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time
import os
 
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
 
 
# 获取mnist数据
mnist_data_set = input_data.read_data_sets('MNIST_data', one_hot=True)
# 声明输入图片数据类型，mnist的手写体图片大小为28*28=784
# None表示行向量的维度是任意的，也就是一次可以输出多张图片
# placeholder 基本都是占位符，先定义，后面会用到的
x = tf.placeholder('float', [None, 784])
# y_为网络的输出,数字十个类别
y_ = tf.placeholder('float', [None, 10])
 
# 把输入的数据转变成二维形式，用于卷积计算
# -1表示一次可以存储多张照片，1表示图像的通道数为1
x_image = tf.reshape(x, [-1, 28, 28, 1])
 
# 卷积核初始化，大小为6个5*5的卷积核，1和x_image的1对应，即为图像的通道数
filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6]))
# 偏置项
bias1 = tf.Variable(tf.truncated_normal([6]))
# 二维卷积计算
conv1 = tf.nn.conv2d(x_image, filter1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.sigmoid(conv1 + bias1)
# 池化层
maxPool2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
filter2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16]))
bias2 = tf.Variable(tf.truncated_normal([16]))
conv2 = tf.nn.conv2d(maxPool2, filter2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.sigmoid(conv2 + bias2)
 
maxPool3 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
filter3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120]))
bias3 = tf.Variable(tf.truncated_normal([120]))
conv3 = tf.nn.conv2d(maxPool3, filter3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3 = tf.nn.sigmoid(conv3 + bias3)
 
# 全连接层，权重初始化
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 120, 80]))
# 偏置项
b_fc1 = tf.Variable(tf.truncated_normal([80]))
# 将卷积的输出展开
h_pool2_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 120])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
 
W_fc2 = tf.Variable(tf.truncated_normal([80, 10]))
b_fc2 = tf.Variable(tf.truncated_normal([10]))
# 输出层，使用softmax进行多分类
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
 
# 损失函数，交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 使用梯度下降法来更新权重，学习速率为0.001，改为0.01的话会导致权重更新有问题，准确率会滴
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
 
sess = tf.InteractiveSession()
 
# 测试准确率，tf.argmax() 计算行或列的最大值，返回最大值下标的向量
# tf.equal() 计算两个向量对应的元素是否相等，返回数据为bool
# tf.cast() 把bool数据转化为浮点型
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
 
# 对所有变量进行初始化
sess.run(tf.global_variables_initializer())
 
start_time = time.time()
for i in range(50000):
    # 获取训练数据，每次100张
    # 我们后面取数据的时候，直接获取feed_dict={x: batch[0], y_true: batch[1]}
    # batch = mnist_data_set.train.next_batch(60)
    batch_xs, batch_ys = mnist_data_set.train.next_batch(100)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    # 每个100次输出当前的准确率
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        # 计算时间间隔
        end_time = time.time()
        print('time: ', (end_time - start_time))
        start_time = end_time
 
print("Test accuracy: {}".format(accuracy.eval(session=sess,
                                               feed_dict={
                                                   x: mnist_data_set.test.images,
                                                   y_: mnist_data_set.test.labels})))
 
# 关闭会话
sess.close()
