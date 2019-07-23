import tensorflow as tf
from PIL import Image
import numpy as np
import os
import model
import pre_process
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def evaluate_one_img():
    test = "D:\新建文件夹\python foot/test/"
    file = os.listdir(test)  # os.listdir()返回指定目录下的所有文件和目录名。
    n = len(file)
    df = pd.read_csv("D:\新建文件夹\python foot/sample_submission.csv")
    for i in range(1,n):
        img_dir = os.path.join(test, file[i])  # 判断是否存在文件或目录name
        image = Image.open(img_dir)
        image = image.resize([208, 208])
        image = np.array(image)
        test_array= image
        #print(test_array.shape)
        with tf.Graph().as_default():#https://www.cnblogs.com/studylyn/p/9105818.html
            BATCH_SIZE = 1
            N_CLASSES = 2
            image = tf.cast(test_array, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image,[1,208,208,3])
        #test, train_labels = pre_process.get_files(test)
        #image, _ = pre_process.get_batch(test, train_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
            logit = model.inference(image, BATCH_SIZE, N_CLASSES)
            logit = tf.nn.softmax(logit)
            x =tf.placeholder(tf.float32, shape =[208,208,3])

            log_test_dir = "D:\新建文件夹\python foot/train_savenet"
            saver = tf.train.Saver()

            with tf.Session() as sess:
                print("从指定路径中加载模型。。。")
            #将模型加载到sess中
                ckpt = tf.train.get_checkpoint_state(log_test_dir)
                if ckpt and ckpt.model_checkpoint_path:#https://blog.csdn.net/u011500062/article/details/51728830/
                    global_step  = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("模型加载成功，训练的步数为 %s", global_step)
                else:
                    print("模型加载失败，文件没有找到。")

            #将图片输入到模型计算
                prediction = sess.run(logit, feed_dict={x: test_array})
                prediction=prediction.clip(min=0.005, max=0.995)
                 # 将图片输入到模型计算
                #print(prediction[:, 1])
                df.set_value(i-1, 'label', prediction[:, 1])
                #print('猫的概率 %.6f' %prediction[:, 0])
                #print('狗的概率 %.6f' %prediction[:, 1])
    df.to_csv('D:\新建文件夹\python foot/pred.csv', index=None)
# 测试

evaluate_one_img()