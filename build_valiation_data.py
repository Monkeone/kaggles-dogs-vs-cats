import random
import shutil
src_dir='D:\新建文件夹\python foot/train/'
target_dir='D:\新建文件夹\python foot/valiation/'
validation_ratio=5
def build_valiation_data(src_dir, target_dir, validation_ratio):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    files = os.listdir(src_dir)
    total_size = len(files)
    validation_size = int(total_size / validation_ratio)
    print(validation_size)
    random.shuffle(files)
    for i in range(validation_size):
        f = files[i]
        shutil.move(os.path.join(src_dir, f), os.path.join(target_dir, f))#移动文件内容，src原地址，target目标地址
    print("total size: {}, validation size: {}".format(total_size, validation_size))
build_valiation_data(src_dir=src_dir,target_dir=target_dir,validation_ratio=validation_ratio)
