# 加入跳跃链接，且跳跃连接的激活函数改为全等映射。
# 深度增加。
# 使用了归一化处理。
import collections
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib import slim
import math

batch_size = 32
num_batches = 100
num_steps_burn_in = 10
total_duration = 0.0
total_duration_squared = 0.0
inputs = tf.random_uniform((batch_size, 224, 224, 3))


class Block(collections.namedtuple("block", ["name", "residual_unit", "args"])):
    """
    作用：描述ResNet块的命名元组.
    用法：Python中存储系列数据，相比与list，tuple中的元素不可修改，在映射中可以当键使用。
    tuple元组的item只能通过index访问，collections模块的namedtuple子类不仅可以使用item的index访问item，
    还可以通过item的name进行访问。
    原型：collections.namedtuple(typename,field_names,verbose,rename)
    """


# 定义卷积操作
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    """
    :param inputs: 输入大小
    :param num_outputs: 通道数
    :param kernel_size: 卷积核大小
    :param stride: 步长
    :param scope:
    :return: 卷积结果
    """
    # 步长为1
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding="SAME", scope=scope)
    # 步长不为1
    else:
        pad_begin = (kernel_size - 1) // 2
        pad_end = kernel_size - 1 - pad_begin

        # pad()函数用于对矩阵进行定制填充
        # 在这里用于对inputs进行向上填充pad_begin行0，向下填充pad_end行0，
        # 向左填充pad_begin行0，向右填充pad_end行0
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding="VALID", scope=scope)


# 定义残差块操作
@slim.add_arg_scope
def residual_unit(inputs, depth, depth_residual, stride, outputs_collections=None,
                  scope=None):
    """

    :param inputs:输入数据
    :param depth: 输入深度 256
    :param depth_residual:输出深度 64
    :param stride:步长1
    :param outputs_collections:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope, "residual_v2", [inputs]) as sc:

        # 输入的通道数
        depth_input = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

        # BN层计算
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope="preac")

        # 全等映射
        # 输入通道数和要输出的通道数一致，则考虑进行降采样操作，
        if depth == depth_input:
            # 如果stride等于1，则不进行降采样操作，
            if stride == 1:
                identity = inputs
            # 如果stride不等于1，则使用max_pool2d，进行步长为stride且池化核为1x1的降采样操作
            else:
                identity = slim.max_pool2d(inputs, [1, 1], stride=stride, scope="shortcut")
        # 输入通道数和要输出的通道数不一致，则使用NIN卷积操作使输入通道数和输出通道数一致
        else:
            identity = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None,
                                   activation_fn=None, scope="shortcut")

        # 卷积操作
        residual = slim.conv2d(preact, depth_residual, [1, 1], stride=1, scope="conv1")
        residual = conv2d_same(residual, depth_residual, 3, stride, scope="conv2")
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None,
                               activation_fn=None, scope="conv3")

        # 将identity的结果和residual的结果相加
        output = identity + residual

        result = slim.utils.collect_named_outputs(outputs_collections, sc.name, output)
        """
        为output的tensor添加别名，并将tensor添加到collections的列表中
        如果这个(列表名，列表)键值对存在于Graph的self._collection字典中，
        则只是在列表上添加，否则会在字典中创建键值对。  
        """

        return result


def resnet_v2(inputs, blocks, num_classes, reuse=None, scope=None):
    with tf.variable_scope(scope, "resnet_v2", [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + "_end_points"

        # 对函数residual_unit()的outputs_collections参数使用参数空间
        with slim.arg_scope([residual_unit], outputs_collections=end_points_collection):

            # 创建ResNet的第一个卷积层和池化层，卷积核大小7x7，深度64，池化核大小3x3
            with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                net = conv2d_same(inputs, 64, 7, stride=2, scope="conv1")
            net = slim.max_pool2d(net, [3, 3], stride=2, scope="pool1")

            # 在两个嵌套的for循环内调用residual_unit()函数堆砌ResNet的结构
            for block in blocks:
                # block.name分别为block1、block2、block3和block4
                with tf.variable_scope(block.name, "block", [net]) as sc:

                    # tuple_value为Block类的args参数中的每一个元组值，
                    # i是这些元组在每一个Block的args参数中的序号
                    for i, tuple_value in enumerate(block.args):
                        # i的值从0开始，对于第一个unit，i需要加1
                        with tf.variable_scope("unit_%d" % (i + 1), values=[net]):
                            # 每一个元组都有3个数组成，将这三个数作为参数传递到Block类的
                            # residual_unit参数中，在定义blockss时，这个参数就是函数residual_unit()
                            depth, depth_bottleneck, stride = tuple_value
                            net = block.residual_unit(net, depth=depth, depth_residual=depth_bottleneck,
                                                      stride=stride)
                    # net就是每一个块的结构
                    net = slim.utils.collect_named_outputs(end_points_collection, sc.name, net)

            # 对net使用slim.batch_norm()函数进行BatchNormalization操作
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope="postnorm")

            # 创建全局平均池化层
            net = tf.reduce_mean(net, [1, 2], name="pool5", keep_dims=True)

            # 如果定义了num_classes，则通过1x1池化的方式获得数目为num_classes的输出
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope="logits")

            return net


def resnet_v2_152(inputs, num_classes, reuse=None, scope="resnet_v2_152"):
    """
    卷积操作的主函数
    :param inputs: 输入数据集的大小
    :param num_classes: 图片的分类结果，代表最后的softmax层有多少节点，也代表label的大小
    :param reuse: 命名空间内的参数是否可以复用
    :param scope: 命名空间的名称
    :return:
    """
    # 有四个ResNet块
    blocks = [
        Block("block1", residual_unit, [(256, 64, 1), (256, 64, 1), (256, 64, 2)]),
        Block("block2", residual_unit, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block("block3", residual_unit, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block("block4", residual_unit, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, reuse=reuse, scope=scope)


# 初始化变量以及常量
def arg_scope(is_training=True, weight_decay=0.0001, batch_norm_decay=0.997,
              batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {"is_training": is_training,
                         "decay": batch_norm_decay,
                         "epsilon": batch_norm_epsilon,
                         "scale": batch_norm_scale,
                         "updates_collections": tf.GraphKeys.UPDATE_OPS}

    with slim.arg_scope([slim.conv2d],
                        # weights_initializer用于指定权重的初始化程序
                        weights_initializer=slim.variance_scaling_initializer(),
                        # weights_regularizer为权重可选的正则化程序
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        # activation_fn用于激活函数的指定，默认的为ReLU函数
                        # normalizer_params用于指定正则化函数的参数
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        # 定义slim.batch_norm()函数的参数空间
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # slim.max_pool2d()函数的参数空间
            with slim.arg_scope([slim.max_pool2d], padding="SAME") as arg_scope:
                return arg_scope


# 定义模型的前向传播过程，这被限制在一个参数空间中
with slim.arg_scope(arg_scope(is_training=False)):
    """
    slim是一种轻量级的tensorflow库，可以使模型的构建，训练，测试都变得更加简单。
    在slim库中对很多常用的函数进行了定义，slim.arg_scope（）是slim库中经常用到的函数之一。
    函数的定义如下；
    用法：def arg_scope(list_ops_or_scope, **kwargs):
    作用：存储给定list_ops集合的默认参数
    这个函数的作用是给list_ops中的内容设置默认值。但是每个list_ops中的每个成员需要用@add_arg_scope修饰才行。
    所以使用slim.arg_scope（）有两个步骤：
    1,用@slim.add_arg_scope修饰目标函数
    2,用slim.arg_scope（）为目标函数设置默认参数.
    """
    # 设置的默认参数是arg_scope函数中is_training=False
    # resnet_v2_152函数需要add_arg_scope装饰，且传入的参数是（inputs, 1000）
    net = resnet_v2_152(inputs, 1000)



init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # 运行前向传播测试过程
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = sess.run(net)
        print(_.shape)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if i % 10 == 0:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    average_time = total_duration / num_batches

    # 打印前向传播的运算时间信息
    print('%s: Forward across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), num_batches, average_time,
           math.sqrt(total_duration_squared / num_batches - average_time * average_time)))
