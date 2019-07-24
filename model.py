import tensorflow as tf

def inference(image, batch_size, n_classes):
    with tf.variable_scope("conv1") as scope:#课本108，variable_scope控制get_variable是获取（reuse=True）还是创建变量
        weights = tf.get_variable("weights", shape=[3,3,3,16], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[16], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(image, weights, strides=[1,1,1,1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75, name="norm1")#局部响应归一化??????
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights", shape=[3,3,16,16], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[16], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope("pooling2_lrn") as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75, name="norm2")
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name="pooling2")

    with tf.variable_scope("local3") as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights", shape=[dim, 128], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope("local4") as scope:
        weights = tf.get_variable("weights", shape=[128, 128], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases,name="local4")

    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights", shape=[128, n_classes], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    softmax_linear = tf.nn.relu(tf.matmul(local4, weights) + biases,name="softmax_linear")

    return softmax_linear

def loss(logits, labels):#输出结果和标准答案
    with tf.variable_scope("loss") as scope:
        cross_entropy= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="entropy_per_example")
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar(scope.name +"/loss",loss)#对标量数据汇总和记录使用tf.summary.scalar
    return loss

def training(loss, learning_rate):
    with tf.name_scope("optimizer"):
        global_step = tf.Variable(0, name="global_step", trainable=False)#定义训练的轮数，为不可训练的参数
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op= optimizer.minimize(loss, global_step=global_step)
        #上两行等价于train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    return train_op

def evalution(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)#下面
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+"/accuracy", accuracy)#用来显示标量信息
    return accuracy

"""
top_1_op取样本的最大预测概率的索引与实际标签对比，top_2_op取样本的最大和仅次最大的两个预测概率与实际标签对比，
如果实际标签在其中则为True，否则为False。
"""