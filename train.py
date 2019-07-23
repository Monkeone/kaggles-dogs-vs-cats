import tensorflow as tf
import numpy as np
import os
import pre_process
import model

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 32
CAPACITY = 256
STEP = 4000   #训练步数应当大于10000
LEARNING_RATE = 0.001

x = tf.placeholder(tf.float32, shape=[None,129792])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

def run_training():
    train_dir = "D:\新建文件夹\python foot/train/"
    log_train_dir = "D:\新建文件夹\python foot/train_savenet/"
    vadiation_dir='D:\新建文件夹\python foot/valiation/'
    train,train_labels = pre_process.get_files(train_dir)
    train_batch, train_label_batch = pre_process.get_batch(train, train_labels, IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
    train_logits= model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss= model.loss(train_logits, train_label_batch)
    train_op = model.training(train_loss, LEARNING_RATE)
    train_acc = model.evalution(train_logits, train_label_batch)
    summary_op = tf.summary.merge_all()#merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。
    # 一般这一句就可显示训练时的各种信息。
    #vadiation, vadiation_labels = pre_process.get_files(vadiation_dir)
    #vadiation_batch, vadiation_label_batch = pre_process.get_batch(vadiation, vadiation_labels, IMG_W,IMG_H,BATCH_SIZE, CAPACITY)
    #vadiation_logits = model.inference(vadiation_batch, BATCH_SIZE, N_CLASSES)
    #vadiation_loss = model.loss(vadiation_logits, vadiation_label_batch)
    #vadiation_acc = model.evalution(vadiation_logits, vadiation_label_batch)
    sess = tf.Session()
    train_writer  =tf.summary.FileWriter(log_train_dir, sess.graph)#指定一个文件用来保存图
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    #  Coordinator  和 start_queue_runners 监控 queue 的状态，不停的入队出队
    coord = tf.train.Coordinator()#https://blog.csdn.net/weixin_42052460/article/details/80714539
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

            if step % 50 == 0:#%.2f表示输出浮点数并保留两位小数。%%表示直接输出一个%
                print("step %d, train loss = %.2f, train accuracy  = %.2f%%" %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)  #?????????????


            if step % 2000 == 0 or (step+1) ==STEP:
                # 每隔2000步保存一下模型，模型保存在 checkpoint_path 中
                print("step %d, vadiation loss = %.2f, vadiation accuracy  = %.2f%%" % (step, vadiation_loss, vadiation_acc * 100.0))
                checkpoint_path = os.path.join(log_train_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
run_training()
