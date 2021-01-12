# _*_coding:utf-8_*_
import tensorflow as tf
 
# 配置神经网络的参数
INPUT_NODE = 784  # 这里输入的参数是图片的尺寸 这里是28*28=784
OUTPUT_NODE = 51  # 这里是输出的图片类型，总共51类
 
IMAGE_SIZE = 28
NUM_CHANNELS = 3
NUM_LABELS = 51
 
# 第一层卷积层的尺寸和深度
CONV1_DEEP = 6
CONV1_SIZE = 5
 
# 第三层卷积层的尺寸和深度
CONV2_DEEP = 16
CONV2_SIZE = 5
 
# 第五层卷积层的尺寸和深度
CONV3_DEEP = 120
CONV3_SIZE = 5
 
# 全连接层的节点个数
FC_SIZE = 84
 
 
# 定义卷积神经网络的前向传播过程
# 这里添加一个新的参数train，用于区分训练过程和测试过程
# 在这个程序中将用到Dropout方法，dropout可以进一步提升模型可靠性，并防止过拟合
# 注意dropout层只能在训练过程中使用
def inference(input_tensor, train, regularizer):
    '''
    声明第一层卷积层的变量并实现前向传播过程，通过使用不同的命名空间来隔离不同层的变量
    这可以让每一层中变量命名只需要考虑在当前层的作用，而不需要担心命名重复的问题
    和标准的LeNet-5模型不太一样，这里定义的卷积层输入为28*28*1的原始MNIST图片箱数
    因为卷积层使用了全0填充，所以输出为28*28*6的矩阵
    :param input_tensor:
    :param train:
    :param regularizer:
    :return:
    '''
    with tf.variable_scope('layer1-conv1'):  # [5,5,3,6]
        conv1_weights = tf.get_variable("weight",
                                        [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [CONV1_DEEP],
                                       initializer=tf.constant_initializer(0.0))
 
        # 使用边长为5， 深度为6的过滤器，过滤器移动的步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights,
                             strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
 
    # 实现第二层池化层的前向传播过程，这里选用最大池化层
    # 池化层过滤器的边长为2，使用全零填充且移动的步长为2，这一层的输入为上一层的输出
    # 也就是28*28*6的矩阵  输出为14*14*6的矩阵
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
 
    # 声明第三层卷积层的变量并实现前向传播过程，这一层输入为14*14*6的矩阵
    # 因为卷积层没有使用全零填充，所以输出为10*10*16的矩阵
    with tf.variable_scope('layer3-conv2'):  # [5,5,3,16]
        conv2_weights = tf.get_variable("weight",
                                        [CONV2_SIZE, CONV2_SIZE, NUM_CHANNELS, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP],
                                       initializer=tf.constant_initializer(0.0))
 
        # 使用边长为5， 深度为16的过滤器，过滤器移动的步长为1，bububu不使用全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights,
                             strides=[1, 1, 1, 1], padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
 
    # 实现第四层池化层的前向传播过程，这一层和第二层的结构是一样的
    # 这里输入10*10*16的矩阵，输出为5*5*16的矩阵
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='VALID')
 
    # 声明第五层全连接层（实际上为卷积层）的变量并实现前向传播过程
    # 这一层输入是5*5*16的矩阵，因为没有使用全0填充，所以输出为1*1*120
    with tf.name_scope('layer5-conv3'):
        conv3_weights = tf.get_variable("weight",
                                        [CONV3_SIZE, CONV3_SIZE, NUM_CHANNELS, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('bias', [CONV3_DEEP],
                                       initializer=tf.constant_initializer(0.0))
 
        # 使用边长为5， 深度为6的过滤器，过滤器移动的步长为1，bububu不使用全0填充
        conv3 = tf.nn.conv2d(pool2, conv3_weights,
                             strides=[1, 1, 1, 1], padding='VALID')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
 
    # 将第五层卷积层的输出转化为第六层全连接层的输入格式
    # 第五层的输出为1*1*120的矩阵，然而第六层全连接层需要的输出格式为向量
    # 所以这里需要将这个1*1*120的矩阵拉直成一个向量
    # relu3.get_shape函数可以得到第五层输出矩阵的维度而不需要手工计算。
    # 注意因为每一层神经网络的输入输出都为一个batch的矩阵，
    # 所以这里得到的维度也包含了一个batch中数据的个数。
    pool_shape = relu3.get_shape().as_list()
 
    # 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长度及深度的乘积
    # 注意这里pool_shape[0]为一个batch中数据的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 通过tf.reshape函数将第五层的输出变成一个batch的向量
    reshaped = tf.reshape(relu3, [pool_shape[0], nodes])
 
    # 声明第六层全连接层的变量并实现前向传播过程，这一层的输入是拉直之后的一组向量
    # 向量的长度为1120，输出是一组长度为84的向量
    # 这一层和之前的LeNet基本一致，唯一的区别就是引入的dropout层
    # dropout在训练时会随机将部分节点的输出改为0
    # dropout可以避免过拟合问题，从而使得在测试数据上的效果更好
    # dropout一般只在全连接层而不是卷积层或者池化层使用
    with tf.variable_scope('layer6-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
 
        fc1_biases = tf.get_variable('bias', [FC_SIZE])
        initializer = tf.constant_initializer(0.1)
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
 
    # 声明第七层全连接的变量并实现前向传播过程，这一层的输入为一组长度为为84的向量
    #输出为51的向量，这一层的输出通过softmax之后就得到了最后的分类结果
    with tf.variable_scope('layer7-fc2'):
        fc2_weights = tf.get_variable('weight',
                                      [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [NUM_LABELS])
        initializer = tf.constant_initializer(0.1)
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
 
    # 返回第七层的输出
    return logit
