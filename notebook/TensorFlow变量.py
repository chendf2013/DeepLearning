import tensorflow as tf

weights = tf.Variable(initial_value=tf.random.normal([3, 4], stddev=1),  # 指定初始化的方法
                      name="weight",  # 参数名称
                      trainable=True,  # 是否需要训练，需要就会加入到GraphKeys.TRAINABLE_VARIABLES中去
                      dtype=tf.float32,  # 指定数据类型
                      validate_shape=True,
                      collections=[tf.GraphKeys.GLOBAL_VARIABLES]
                      )
init_op = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session()as sess:
    sess.run(init_op)
    weg = sess.run(weights)
    print(weg)
