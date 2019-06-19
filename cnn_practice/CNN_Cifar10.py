import Cifar10_data
import tensorflow as tf
import numpy as np
import time
import math

max_steps = 4000
batch_size = 128
num_examples_for_eval = 10000
data_dir = "C:/Users/xiaomi/Desktop/DeepLearning/Cifar_CNN_tensorflow/cifar-10-batches-bin"


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        # multiply()函数原型multiply(x,y,name)
        # l2_loss()函数原型l2_loss(t,name)
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)
    return var


# 对于用于训练的图片数据，distorted参数为True，表示进行数据增强处理
images_train, labels_train = Cifar10_data.inputs(data_dir=data_dir,
                                                 batch_size=batch_size, distorted=True)

# 对于用于训练的图片数据，distorted参数为Nnone，表示不进行数据增强处理
images_test, labels_test = Cifar10_data.inputs(data_dir=data_dir,
                                               batch_size=batch_size, distorted=None)

# 创建placeholder
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])

# 第一个卷积层
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding="SAME")

# 第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding="SAME")

# 拉直数据
# reshape()函数原型reshape(tensor,shape,name)
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value

# 第一个全连层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 第二个全连层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 第三个全连层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
fc_bias3 = tf.Variable(tf.constant(0.0, shape=[10]))
result = tf.add(tf.matmul(local4, weight3), fc_bias3)

# 计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,
                                                               labels=tf.cast(y_, tf.int64))
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 函数原型in_top_k(predictions,targets,k,name)
top_k_op = tf.nn.in_top_k(result, y_, 1)


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 开启多线程
    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch,
                                                              y_: label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)

            # 打印每一轮训练的耗时
            print("step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)" % (
            step, loss_value, examples_per_sec, sec_per_batch))

    # math.ceil()函数用于求整，原型为ceil(x)
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    true_count = 0
    # total_sample_count = num_iter * batch_size
    total_sample_count = num_batch * batch_size

    # 在一个for循环内统计所有预测正确的样例的个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch,
                                                      y_: label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    # print("accuracy = %.3f%%" % ((true_count/total_sample_count)*100))
    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
