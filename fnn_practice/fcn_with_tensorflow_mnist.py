import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# read_data_sets()从原始的数据包中解析成训练禾测试神经网络时使用的格式
# 如果在传入的路径中没有找到MINST数据集文件，会自动调用其他其他函数自动下载
# one_hot=True 是否将样本图片对应到标注信息（）
mnist = input_data.read_data_sets("C:/Users/xiaomi/Desktop/DeepLearning/Minist_TensorFlow",
                                  one_hot=True)

batch_size = 100  # 设置每一轮训练的batch大小
learning_rate = 0.9  # 学习率
learning_rate_decay = 0.999  # 学习率的衰减
max_steps = 30000  # 最大训练步数

# 定义存储训练轮数的变量，在使用Tensorflow训练神经网络时，
# 一般会将代表训练轮数的变量通过trainable参数设置为不可训练的
training_step = tf.Variable(0, trainable=False)


# 定义得到隐藏层和输出层的前向传播计算方式，激活函数使用ReLU()
def hidden_layer(input_tensor, weights1, biases1, weights2, biases2, layer_name):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2


# 定义输入层位置，大小为（n,784），placeholder解决了在有限的输入节点上实现高效地接收大量数据的问题，在运行会话时（Session.run()）,通过feed_dict字典传入值
x = tf.placeholder(tf.float32, [None, 784], name="x-input")  # INPUT_NODE=784

# 定义标签层位置，大小为（n,10）
y_ = tf.placeholder(tf.float32, [None, 10], name="y-output")  # OUT_PUT=10

# 生成隐藏层权重参数（随机数），大小为（784，500），标准差为0.1，均值为0的随机数矩阵
weights1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))

# 生成隐藏层截距项参数（常数），大小为（500，）值为0.1
biases1 = tf.Variable(tf.constant(0.1, shape=[500]))

# 生成输出层权重参数，大小为（500，10）
weights2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))

# 生成输出层截距项参数
biases2 = tf.Variable(tf.constant(0.1, shape=[10]))

# 计算经过神经网络前向传播后得到的y的值
y = hidden_layer(x, weights1, biases1, weights2, biases2, 'y')

# 滑动平均(exponential moving average)，或者叫做指数加权平均(exponentially weighted moving average)，
# 可以用来估计变量的局部均值，使得变量的更新与一段时间内的历史取值有关。
# 滑动平均的好处：占内存少，不需要保存过去10个或者100个历史θ值，就能够估计其均值。
# 在采用随机梯度下降算法训练神经网络时提高最终模型在测试数据上的表现，使用滑动平均模型
# 参数decay 用于控制模型更新的速度，值越大模型越趋于稳，一般设置为非常接近1的数（0.99或0.999）
# 参数num_updates 限制decay的大小，值越小，会得到较快的影子变量更新速度，此处设置为当前网络的训练轮数
averages_class = tf.train.ExponentialMovingAverage(0.99, training_step)

# 通过给apply函数提供要进行滑动平均计算的变量，执行影子变量的计算
# 参数 var_list 传递进来的参数列表
# train_variables()函数返回集合图上Graph.TRAINABLE_VARIABLES中的元素，这个集合的元素就是所有没有指定trainable_variables=False的参数
averages_op = averages_class.apply(tf.trainable_variables())

# 再次计算经过神经网络前向传播后得到的y的值，这里使用了滑动平均，但要牢记滑动平均值只是一个影子变量
average_y = hidden_layer(x, averages_class.average(weights1), averages_class.average(biases1),
                         averages_class.average(weights2), averages_class.average(biases2), 'average_y')

# 计算交叉熵损失的函数原型为
# sparse_softmax_cross_entropy_with_logits(_sential, labels, logdits, name)
# 适用于每个类别相互独立且排斥的情况，即一幅图只能属于一类。
# 参数logits 是神经网络不包括softmax层的前向传播结果，是一个batch_size * 10(batch_size行，10列)的二维数组，每一行表示一个样例前向传播的结果，
# 参数lables 训练数据的正确答案
# argmax()函数原型为argmax(input, axis, name, dimension),axis参数“1”表示在每一行选取最大值对应的下标，表示了每一个样例对应的数字识别结果
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

regularizer = tf.contrib.layers.l2_regularizer(0.0001)  # 计算L2正则化损失函数
regularization = regularizer(weights1) + regularizer(weights2)  # 计算模型的正则化损失
loss = tf.reduce_mean(cross_entropy) + regularization  # 总损失

# 用指数衰减法设置学习率，这里staircase参数采用默认的False，即学习率连续衰减
laerning_rate = tf.train.exponential_decay(learning_rate, training_step, mnist.train.num_examples / batch_size,
                                           learning_rate_decay)
# 使用GradientDescentOptimizer优化算法来优化交叉熵损失和正则化损失
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=training_step)

# 在训练这个模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数
# 又需要更新每一个参数的滑动平均值，control_dependencies()用于完成这样的一次性多次操作，
# 同样的操作也可以使用下面这行代码完成：
# train_op = tf.group(train_step,averages_op)
with tf.control_dependencies([train_step, averages_op]):
    train_op = tf.no_op(name="train")

# 检查使用了滑动平均值模型的神经网络前向传播结果是否正确。
# equal()函数原型为equal(x, y, name)，用于判断两个张量的每一维是否相等，如果相等返回True,否则返回False。
crorent_predicition = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

# cast()函数原型为cast(x, DstT, name)。在这里用于将一个bool型的数据转为float32类型
# 之后会将得到的float32 的数据求一个平均值，这个平均值就是模型在这一组数据上的正确率
accuracy = tf.reduce_mean(tf.cast(crorent_predicition, tf.float32))

with tf.Session() as sess:
    # 函数初始化全部变量
    tf.global_variables_initializer().run()

    # 准备验证数据，
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    # 准备测试数据，
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    for i in range(max_steps):
        if i % 1000 == 0:
            # 循环的轮数是1000的倍数时，执行滑动平均模型模型结果在验证数据集上的验证。
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d trainging step(s) ,validation accuracy"
                  "using average model is %g%%" % (i, validate_accuracy * 100))

        # 产生这一轮使用的一个batch的训练数据，并进行训练
        # input_data.read_data_sets()函数生成的类提供了train.next_bacth()函数，获取一个batch的训练数据，并feed给x，y_
        # 通过设置函数的batch_size参数就可以从所有的训练数据中提读取一小部分作为一个训练batch
        xs, ys = mnist.train.next_batch(batch_size=100)
        sess.run(train_op, feed_dict={x: xs, y_: ys})

    # 使用测试数据集检验神经网络训练之后的最终正确率
    # 为了能得到百分数输出，需要将得到的test_accuracy扩大100倍
    # Session.run()用于运行会话（计算张量的取值），可以传入多个需要计算的张量
    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("After %d trainging step(s) ,test accuracy using average"
          " model is %g%%" % (max_steps, test_accuracy * 100))
