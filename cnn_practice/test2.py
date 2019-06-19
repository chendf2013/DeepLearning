
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("C:\\Users\\xiaomi\\Desktop\\DeepLearning\\data\\minist-10-batches", one_hot=True)
x_test, y_test = mnist.test.next_batch(300)
print(y_test.dtype)

# 计算当训练模型的输出的准确率
# 验证集的输出
# y_practice = tf.argmax(average_y, 1)
"""
返回一个张量在轴上的最大值的索引。
"""
# 验证集的标签
# y_label_num = tf.argmax(y_label, 1)
"""
tf.argmax是tensorflow用numpy的np.argmax实现的，
它能给出某个tensor对象在某一维上的其数据最大值所在的索引值，
常用于metric（如acc）的计算。
与arg_max()功能一致
"""
# 输出与标签比对
# prediction = tf.equal(y_practice, y_label_num)
# print(prediction)
"""
tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，
反正返回False，返回的值的矩阵维度和A是一样的
"""
# 将比对结果转化为百分数
# accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
"""

tf.reduce_mean(input_tensor)
计算元素跨张量维数的平均值
因为本此计算的结果是0，1，因此平均值就是正确率
"""

