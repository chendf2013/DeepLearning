# 使用了模型迁移
# 加入LRN层（被证实没啥用处）
# 使用Inception模块
# 平均池化和最大池化都运用了
# 加入了BN层


import tensorflow as tf
import os.path
import random
import json
import numpy as np
from tensorflow.python.platform import gfile

assembled_data_dict = "C:\\Users\\xiaomi\\Desktop\\DeepLearning\\data\\assembled_flower_data"
# https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz 处下载训练的图片集合
input_data = "C:\\Users\\xiaomi\\Desktop\\DeepLearning\\data\\flower_photos"
# http://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip 处下载的inception-v3模型已经训练好的参数
inceptonV3_model = "C:\\Users\\xiaomi\\Desktop\\DeepLearning\\data\\inception-v3_dec_2015\\tensorflow_inception_graph.pb"

num_steps = 20
BATCH_SIZE = 50
bottleneck_size = 2048  # InceptionV3模型瓶颈层的节点个数
num_classes = 5


# 按花名收集所有图片的路径，并分为三个数据集，分别是test,vail,train
def get_input_data(input_data_dict):
    """
    将下载的图像按类别和训练集，测试集，验证集进行分类保存路径
    :param input_data_dict: 下载的图像集合
    :return: 图像路径字典
    """
    result = dict()
    # 用于保存最终图片结果的路径，格式如下
    """
{
    "roses": {
                "training": training_images_path_list,
                "testing": testing_images_path_list,
                "validation": validation_images_path_list
            },
    "sunflowers": {
                "training": training_images_path_list,
                "testing": testing_images_path_list,
                "validation": validation_images_path_list
                    },
    "tulips": {
                "training": training_images_path_list,
                "testing": testing_images_path_list,
                "validation": validation_images_path_list
                },
    "dandelion": {
                "training": training_images_path_list,
                "testing": testing_images_path_list,
                "validation": validation_images_path_list
            },
    "daisy": {
                "training": training_images_path_list,
                "testing": testing_images_path_list,
                "validation": validation_images_path_list
            }}
"""
    # 获取花朵文件夹下的花朵类别子文件夹列表
    path_list_all = [x[0] for x in os.walk(input_data_dict)]
    """os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下,包含当前文件夹路径，因此需要去掉"""
    path_list_all = path_list_all[1:]
    all_flower_category = list()
    for path_list_first in path_list_all:
        flower_category_dict = dict()
        training_images_path_list = []
        testing_images_path_list = []
        validation_images_path_list = []
        path_list = [x[2] for x in os.walk(path_list_first)][0]
        for path in path_list:
            path = os.path.join(path_list_first, path)
            random_int = random.randint(1, 100)
            if random_int <= 10:
                validation_images_path_list.append(path)
            elif random_int <= 30:
                testing_images_path_list.append(path)
            elif random_int <= 100:
                training_images_path_list.append(path)
        flower_category_dict["training"] = training_images_path_list
        flower_category_dict["testing"] = testing_images_path_list
        flower_category_dict["validation"] = validation_images_path_list
        flower_category = os.path.basename(path_list_first)
        result.update({flower_category: flower_category_dict})
        # 写入花的类别
        all_flower_category.append(flower_category)
        with open(os.path.join(assembled_data_dict, "all_flower_category.txt"), "w") as f:
            f.write(json.dumps(all_flower_category))

    return result


# 按照测试，验证，训练集持久化所有数据
def persistence_data(flower_dict):
    picture_dict = dict()
    """
    {
    "testing":[
            {'dandelion': 'C:\\Users\\xiaomi\\Desktop\\DeepLearning\\data\\flower_
            photos\\dandelion\\14373114081_7922bcf765_n.jpg'},
            {}],
    "training":[{},{}],
    "validation":[{},{}],
    }
    """
    testing_reset = list()
    training_reset = list()
    validation_reset = list()
    for flower_category, data_reset in flower_dict.items():
        for reset_name, path_list in data_reset.items():
            for path in path_list:
                if reset_name == "training":
                    training_reset.append({flower_category: path})
                elif reset_name == "testing":
                    testing_reset.append({flower_category: path})
                elif reset_name == "validation":
                    validation_reset.append({flower_category: path})
    picture_dict["training"] = training_reset
    picture_dict["testing"] = testing_reset
    picture_dict["validation"] = validation_reset
    # print(len(validation_reset))
    for k, v in picture_dict.items():
        story_path = os.path.join(assembled_data_dict, k + ".txt")
        if not os.path.exists(story_path):
            os.makedirs(story_path)
        with open(story_path, "w") as f:
            f.write(json.dumps(v))


# 获取喂入的一个batch大小的图片列表
def get_picture_list_of_input(record_set_category):
    # 初始化保存结果的数据类型
    all_data_list = list()

    # 获取所有的数据
    with open(os.path.join(assembled_data_dict, record_set_category + ".txt"), "r") as f:
        all_picture = f.read()
        all_picture_path = json.loads(all_picture)
        all_picture_num = len(all_picture_path)
        print("第一次", all_picture_num)

    # 如果数据不够了，重新加载数据
    if all_picture_num < BATCH_SIZE:
        persistence_data(get_input_data(input_data))
        with open(os.path.join(assembled_data_dict, record_set_category + ".txt"), "r") as f:
            all_picture = f.read()
            all_picture_path = json.loads(all_picture)
            all_picture_num = len(all_picture_path)
            print("重新加载", all_picture_num)

    # 获取一个batch
    n = 0
    for i in range(BATCH_SIZE):
        index = random.randint(0, all_picture_num - 1 - n)
        n += 1
        # print(index)
        all_data_list.append(all_picture_path.pop(index))

    # 剔除一个batch的数据后重新写入
    with open(os.path.join(assembled_data_dict, record_set_category + ".txt"), "w") as f:
        f.write(json.dumps(all_picture_path))
    # print(all_data_list)
    return all_data_list


# 获取喂入的一个batch大小的向量列表
def get_data_list_of_input(session, record_set_category,
                           jpeg_data_tensor, bottleneck_tensor):
    # 初始化保存结果的数据类型
    feature_vector_list = list()
    label_list = list()

    # 获取所有花的类别
    with open(os.path.join(assembled_data_dict, "all_flower_category.txt"), "r") as f:
        all_flower_category = json.loads(f.read())
        num_of_category = len(all_flower_category)
        # print(num_of_category)
    # 获取图片列表
    picture_list = get_picture_list_of_input(record_set_category)

    for picture in picture_list:
        for k, v in picture.items():
            # 获取label
            index = all_flower_category.index(k)
            label = np.zeros(num_of_category, dtype=np.float32)
            label[index] = 1.0
            label_list.append(label)
            # 获取input
            data = gfile.FastGFile(v, "rb").read()
            bottleneck_values = session.run(bottleneck_tensor, feed_dict={jpeg_data_tensor: data})
            bottleneck_values = np.squeeze(bottleneck_values)
            feature_vector_list.append(bottleneck_values)

    return feature_vector_list, label_list


# 读取已经训练好的Inception-v3模型。
with gfile.FastGFile(inceptonV3_model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# 使用import_graph_def()函数加载读取的InceptionV3模型后会返回
# 图像数据输入节点的张量名称以及计算瓶颈结果所对应的张量，函数原型为
# import_graph_def(graph_def,input_map,return_elements,name,op_dict,producer_op_list)
bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def,
                                                          return_elements=["pool_3/_reshape:0",
                                                                           "DecodeJpeg/contents:0"])
x = tf.placeholder(tf.float32, [None, bottleneck_size], name='BottleneckInputPlaceholder')
y_ = tf.placeholder(tf.float32, [None, num_classes], name='GroundTruthInput')

# 定义一层全连接层
with tf.name_scope("final_training_ops"):
    weights = tf.Variable(tf.truncated_normal([bottleneck_size, num_classes], stddev=0.001))
    biases = tf.Variable(tf.zeros([num_classes]))
    logits = tf.matmul(x, weights) + biases
    final_tensor = tf.nn.softmax(logits)

# 定义交叉熵损失函数以及train_step使用的随机梯度下降优化器
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_mean)

# 定义计算正确率的操作
correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(y_, 1))
evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
if __name__ == '__main__':
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # data_list = get_data_list_of_input(sess, "validation", jpeg_data_tensor, bottleneck_tensor)
        for i in range(num_steps):
            "session, image_path_dict, batch_size, record_set_category,jpeg_data_tensor, bottleneck_tensor"
            train_bottlenecks, train_labels = get_data_list_of_input(sess, "training",
                                                                     jpeg_data_tensor, bottleneck_tensor)
            # print(len(train_bottlenecks))
            # print(len(train_labels))
            sess.run(train_step, feed_dict={x: train_bottlenecks, y_: train_labels})

            if i % 10 == 0:
                validation_bottlenecks, validation_labels = get_data_list_of_input(sess, "validation",
                                                                                   jpeg_data_tensor,
                                                                                   bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    x: validation_bottlenecks,
                    y_: validation_labels})
                print("Step %d: Validation accuracy = %.1f%%" % (i, validation_accuracy * 100))
        test_bottlenecks, test_labels = get_data_list_of_input(sess,
                                                               "testing",
                                                               jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={x: test_bottlenecks,
                                                             y_: test_labels})
        print("Finally test accuracy = %.1f%%" % (test_accuracy * 100))

"""
第一次 997
第一次 150
Step 0: Validation accuracy = 22.0%
第一次 947
第一次 897
第一次 847
第一次 797
第一次 747
第一次 697
第一次 647
第一次 597
第一次 547
第一次 497
第一次 100
Step 10: Validation accuracy = 64.0%
第一次 447
第一次 397
第一次 347
第一次 297
第一次 247
第一次 197
第一次 147
第一次 97
第一次 47
重新加载 2556
第一次 705
Finally test accuracy = 78.0%
"""
