import math
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("C:\\Users\\xiaomi\\Desktop\\DeepLearning\\data\\minist-10-batches", one_hot=True)

# 初始化超参数
learning_rate = 0.01  # 学习率
learning_rate_decay = 0.99  # 衰减系数
max_learning_step = 30000  # 最大训练次数
batch_size = 100  # batch大小
regular_penalty_coefficient = 0.0001  # 正则惩罚项系数
fc_1_unit = 512
fc_2_unit = 10
exponential_moving_average_decay = 0.99  # 滑动平均的初始衰减系数

# 初始化变量
x = tf.placeholder(name="x_input", shape=([batch_size, 28, 28, 1]), dtype=tf.float32)  # x
"""x,y应使用placeholder而不是使用get_variable"""
y_label = tf.placeholder(name="y_label", shape=([batch_size, 10]), dtype=tf.float32)  # y^
regular = tf.contrib.layers.l2_regularizer(regular_penalty_coefficient)  # 使用L2惩罚项惩罚
training_step = tf.Variable(0, trainable=False)  # 反向传播优化过程中的步数
learning_rate = tf.train.exponential_decay(learning_rate,
                                           training_step,
                                           mnist.train.num_examples / batch_size,
                                           learning_rate_decay,
                                           staircase=True)  # 学习率指数衰减


# 定义前向传播过程
def hidden(train_data, reg, averages, reuse):
    """
    前向传播：卷积层-->池化层-->卷积层-->池化层-->归整-->全连接层-->全连接层
    :param
    train_data: 训练集（batch_size,28,28,1）
    reg：正则惩罚项方法
    :return: 预测值 shape = batch_size*10
    """
    # 卷积层一
    # reuse=True
    with tf.variable_scope("con_1", reuse=reuse):
        """
        tf.get_variable(name,  shape, initializer): name就是变量的名称，shape是变量的维度，
        initializer是变量初始化的方式，初始化的方式有以下几种：
        tf.constant_initializer：常量初始化函数
        tf.random_normal_initializer：正态分布
        tf.truncated_normal_initializer：截取的正态分布
        tf.random_uniform_initializer：均匀分布
        tf.zeros_initializer：全部是0
        tf.ones_initializer：全是1
        tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值
        """
        con_1_filter = tf.get_variable(name="filter", shape=([5, 5, 1, 32]), dtype=float,
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        con_1_biases = tf.get_variable(name="biases", shape=([32]), dtype=float,
                                       initializer=tf.constant_initializer(0.0))
        """维度是（32），不是（32，1）"""
        con_1_res_1 = tf.nn.conv2d(input=train_data, filter=con_1_filter, strides=[1, 1, 1, 1], padding="SAME",
                                   use_cudnn_on_gpu=True,
                                   data_format="NHWC", dilations=[1, 1, 1, 1], name="con_1")
        """
        input, filter, strides, padding, use_cudnn_on_gpu=True, 
        data_format="NHWC", dilations=[1, 1, 1, 1], name=None
        """
        con_1_res_2 = tf.nn.bias_add(value=con_1_res_1,
                                     bias=con_1_biases,
                                     data_format=None,
                                     name=None)
        con_1_res_3 = tf.nn.relu(features=con_1_res_2,
                                 name=None)
    # 池化层二
    with tf.variable_scope("pooling_2", ):
        """ksize=([2,2])-->ksize=([1,2,2,1])"""
        pooling_2_res = tf.nn.max_pool(value=con_1_res_3, ksize=([1, 2, 2, 1]), strides=[1, 2, 2, 1], padding="SAME",
                                       data_format="NHWC", name=None)
        """value, ksize, strides, padding, data_format = "NHWC", name = None"""

    # 卷积层三
    with tf.variable_scope("con_3", reuse=reuse):
        con_3_filter = tf.get_variable(name="filter", shape=([5, 5, 32, 64]), dtype=float,
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        con_3_biases = tf.get_variable(name="biases", shape=([64]), dtype=float,
                                       initializer=tf.constant_initializer(0.0))
        con_3_sres_1 = tf.nn.conv2d(input=pooling_2_res, filter=con_3_filter, strides=[1, 1, 1, 1], padding="SAME",
                                    use_cudnn_on_gpu=True,
                                    data_format="NHWC", dilations=[1, 1, 1, 1], name="con_1")
        con_3_res_2 = tf.nn.bias_add(value=con_3_sres_1,
                                     bias=con_3_biases,
                                     data_format=None,
                                     name=None)
        con_3_res_3 = tf.nn.relu(features=con_3_res_2,
                                 name=None)
    # 池化层四
    with tf.variable_scope("pooling_4", ):
        pooling_4_res = tf.nn.max_pool(value=con_3_res_3, ksize=([1, 2, 2, 1]), strides=[1, 2, 2, 1], padding="SAME",
                                       data_format="NHWC", name=None)
    # 归整(300, 7, 7, 64)-->(300,3136)
    shape = pooling_4_res.get_shape().as_list()
    """
    get_shape()函数可以得到这一层维度信息，由于每一层网络的输入输出都是一个batch的矩阵，
    所以通过get_shape()函数得到的维度信息会包含这个batch中数据的个数信息
    shape[1]是长度方向，shape[2]是宽度方向，shape[3]是深度方向
    shape[0]是一个batch中数据的个数，reshape()函数原型reshape(tensor,shape,name)
    """
    size_of_feature_map = shape[1] * shape[2] * shape[3]
    # 现在是tensor类型，不是numpy类型，因此使用tf.reshape 而不是np.reshape
    pooling_4_res = tf.reshape(pooling_4_res, [batch_size, size_of_feature_map])

    # 全连接层五
    with tf.variable_scope("fc_5", reuse=reuse):
        fc_5_weights = tf.get_variable(name="fc_5_weights", shape=([size_of_feature_map, fc_1_unit]), dtype=float,
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 计算正则惩罚项
        tf.add_to_collection("losses", reg(fc_5_weights))

        fc_5_biases = tf.get_variable(name="fc_5_biases", shape=([fc_1_unit]), dtype=float,
                                      initializer=tf.constant_initializer(0.0))
        # 训练集没有影子变量
        if averages is None:
            fc_5_res_1 = tf.matmul(pooling_4_res, fc_5_weights) + fc_5_biases
            fc_5_res_2 = tf.nn.relu(features=fc_5_res_1)
        # 验证集有影子变量
        else:
            fc_5_res_1 = tf.matmul(pooling_4_res, averages.average(fc_5_weights)) + averages.average(fc_5_biases)
            fc_5_res_2 = tf.nn.relu(features=fc_5_res_1)

    # 全连接层六
    with tf.variable_scope("fc_6", reuse=reuse):
        fc_6_weights = tf.get_variable(name="fc_6_weights", shape=([fc_1_unit, fc_2_unit]), dtype=float,
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", reg(fc_6_weights))
        fc_6_biases = tf.get_variable(name="fc_6_biases", shape=([fc_2_unit]), dtype=float,
                                      initializer=tf.constant_initializer(0.0))
        if averages is None:

            fc_6_res_1 = tf.matmul(fc_5_res_2, fc_6_weights) + fc_6_biases
            """
            tensorflow中将sigmoid激活函数和softmax回归放到一个函数中，因此此处不进行激活处理，直接使用softmax求交叉熵函数
            """
        else:
            fc_6_res_1 = tf.matmul(fc_5_res_2, averages.average(fc_6_weights)) + averages.average(fc_6_biases)

    return fc_6_res_1


# 数据集验证
def val_right(y_, y_real):
    top_k_op = tf.nn.in_top_k(y_, tf.argmax(tf.cast(y_real, tf.int32), 1), 1)
    return top_k_op


# 求取损失函数
y = hidden(x, regular, None, reuse=False)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_label, 1))  # 交叉熵
cross_entropy_mean = tf.reduce_mean(cross_entropy)
regular_penalty_loss = tf.add_n(tf.get_collection('losses'))  # 正则惩罚项
loss = cross_entropy_mean + regular_penalty_loss
# 反向传播优化器
train_step = tf.train.GradientDescentOptimizer(learning_rate). \
    minimize(loss, global_step=training_step)

# 滑动求取模型参数的平均值，用来验证参数
# 初始化滑动平均类
average_class = tf.train.ExponentialMovingAverage(exponential_moving_average_decay, training_step)
# 计算所有可训练的参数的影子变量
averages_op = average_class.apply(tf.trainable_variables())
# 计算当前训练的模型的输出
average_y = hidden(x, regular, average_class, reuse=True)

# 验证集验证
val_accuracy = val_right(average_y, y_label)

# 验证集验证
test_accuracy = val_right(y, y_label)
"""
tf.cast(x,dtype,name=None)
将x的数据格式转化成dtype数据类型.
例如，原来x的数据格式是bool， 那么将其转化成float以后，
就能够将其转化成0和1的序列。反之也可以.
tf.nn.in_top_k组要是用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量，
tf.nn.in_top_k(prediction, target, K):prediction就是表示你预测的结果，
大小就是预测样本的数量乘以输出的维度，类型是tf.float32等。target就是实际样本类别的标签，
大小就是样本数量的个数。K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值。
一般都是取1。
"""

# 规定计算顺序
with tf.control_dependencies([train_step, val_accuracy]):
    """
    先进行反向传播训练，再进行验证集验证
    """
    train_op = tf.no_op(name="train")
if __name__ == '__main__':
    with tf.Session() as sess:
        # 必须初始化参数
        tf.global_variables_initializer().run()
        # 进行训练
        for train_steps in range(max_learning_step):
            # 获取训练集batch
            x_train, y_train = mnist.train.next_batch(batch_size)
            """
            拿到的数据维度是（batch_size,784）-->(batch_size,28,28,1)
            """
            # 对数据进行处理
            x_train = np.reshape(x_train, (batch_size, 28, 28, 1))
            # 开始训练
            sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
            # 每一百次进行验证
            if train_steps % 500 == 0:
                # 获取验证集batch
                x_val, y_val = mnist.validation.next_batch(batch_size)
                x_val = np.reshape(x_val, (batch_size, 28, 28, 1))
                validate_feed = {x: x_val, y_label: y_val}
                accuracy_rate = sess.run(val_accuracy, feed_dict=validate_feed)
                accuracy_rate = np.sum(accuracy_rate)
                print(
                    "After a {} training sessions,the accuracy_rate is "
                    "{}".format(str(train_steps), str(accuracy_rate)))
        # 训练完毕，测试集测试
        # 获取验证集图片个数
        num_of_test_pic = mnist.test.images.shape[0]
        # 计算需要验证的轮数
        times_of_test = int(math.ceil(num_of_test_pic / batch_size))
        """
        # times_of_test = num_of_test_pic/batch_size
        math.ceil()函数用于求整，原型为ceil(x)
        """
        # 初始化正确率变量
        true_count = 0
        for time in range(times_of_test):
            x_test, y_test = mnist.test.next_batch(batch_size)
            x_test = np.reshape(x_test, (batch_size, 28, 28, 1))
            test_feed = {x: x_test, y_label: y_test}
            count = sess.run(test_accuracy, feed_dict=test_feed)
            true_count += np.sum(count)
        print("At last,the accuracy_rate is {}".format(str((true_count / num_of_test_pic) * 100)))
