import numpy as np
import tensorflow as tf
import os


def get_files(file_dir):
    cats = []
    dogs = []
    cats_label = []
    dogs_label = []
    img_dirs = os.listdir(file_dir)#读取文件名下所有！目录名(列表形式)
    for img_name in img_dirs:# cat.0.jpg
        name = img_name.split(".")# ['cat', '0', 'jpg']
        if  name[0] == "cat":
            cats.append(file_dir + img_name)#此处不可以省为img_name,下个函数tf.train.slice_input_producer读取的是地址！！
            cats_label.append(0)
        else:
            if name[0] == "dog":
                dogs.append(file_dir + img_name)
                dogs_label.append(1)

    img_list = np.hstack((cats, dogs))#列表（字符串形式）
    label_list = np.hstack((cats_label, dogs_label))#列表（整数形式）
    return img_list, label_list

#############################################

def get_batch(image, label, image_w, image_h, batch_size, capacity):#capacity: 队列中 最多容纳图片的个数

    input_queue = tf.train.slice_input_producer([image, label])#tf.train.slice_input_producer是一个tensor生成器，作用是
    # 按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
    label = input_queue[1]
    img_contents = tf.read_file(input_queue[0])#一维
    image = tf.image.decode_jpeg(img_contents, channels=3)#解码成三维矩阵
    image = tf.image.resize_image_with_crop_or_pad(image, image_w, image_h)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)

    # 生成批次  num_threads 有多少个线程根据电脑配置设置
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)

    return image_batch, label_batch