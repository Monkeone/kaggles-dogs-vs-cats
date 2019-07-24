import os
import tensorflow as tf
import numpy as np
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def get_labels(file_dir):
    label_cats = []
    label_dogs = []
    # 载入数据路径并写入标签值
    m=os.listdir(file_dir)
    #n=len(m)
    for file in m:
        name = file.split(sep='.')
        # name的形式为['dog', '9981', 'jpg']
        # os.listdir将名字转换为列表表达
        if name[0] == 'cat':
            # 注意文件路径和名字之间要加分隔符，不然后面查找图片会提示找不到图片
            # 或者在后面传路径的时候末尾加两//  'D:/Python/neural network/Cats_vs_Dogs/data/train//'
            label_cats.append(0)
        else:
            #dogs.append(file_dir + file)
            label_dogs.append(1)
            # 猫为0，狗为1
    # 打乱文件顺序
    label_list = np.hstack((label_cats, label_dogs))
    # np.hstack()方法将猫和狗图片和标签整合到一起,标签也整合到一起
    temp = np.array([label_list])
    temp = temp.transpose()  # 转置
    # 将其转换为10行1列，第一列是label_list的数据
    label_list = list(temp[:, 0])  # 取所有行的第1列数据，并转换为int
    label_list = [int(i) for i in label_list]
    label_list=label_list[0:1200:1]
    label_list=np.array(label_list)
    label_list=label_list.reshape((1,1200))
    #label_list = label_list.reshape((2000, 1))
    #print(label_list)
    return  label_list
# 图片数组
def get_image(test, image_w, image_h):
    image_list=[]
    file = os.listdir(test)
    n=len(file)
    #print(str(image[0]))
    for i in range(0,1200):
        img_dir = os.path.join(test, str(file[i]))
        image = Image.open(img_dir)
        image = image.resize([208, 208])
        image = np.array(image)
        image_list.append(image)
    image=np.array(image_list)
    #print(image.shape)
    return image
