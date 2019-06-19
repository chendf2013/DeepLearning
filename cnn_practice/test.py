import tensorflow as tf

list_a = [[1,2,3,4,5],
          [3,3,3,1,6],
          [5,1,2,1,1]]

sess = tf.InteractiveSession()

argmax0 = tf.arg_max(list_a, 0)
print("argmax 0={}".format(argmax0.eval()))
argmax1 = tf.arg_max(list_a, 1)
print("argmax 1={}".format(argmax1.eval()))

argmax2 = tf.argmax(list_a, 0)
print("argmax 0={}".format(argmax0.eval()))
argmax3 = tf.argmax(list_a, 1)
print("argmax 1={}".format(argmax1.eval()))
# 输出
# argmax 0=[2 1 0 0 1]
# argmax 1=[4 4 0]