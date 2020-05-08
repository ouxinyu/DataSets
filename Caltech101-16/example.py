'''
  @Author: Xinyu Ou
  @Date:   2020-05-07 11:04:20
  @Last Modified by:   Your name
  @Last Modified time: 2020-05-07 11:04:20
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def transform_img(img, augment=True):
    # 数据预处理
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224, 224))
    # img = cv2.resize(img, (28, 28))
    # if augment is True:
    #     img = image_augment(img)
    # show pic
    plt.imshow(img)
    print(img.shape)
    plt.show()

    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    # img = np.mean(img, axis = 0).reshape((1, 28, 28))
    return img


def t_loader(dataDir="Caltech101-16/", batch_size=8):
    # 载入训练集
    # 将datadir目录下的文件列出来，每条文件都要读入
    fileNames = np.loadtxt(dataDir+"Train.txt", dtype=np.str)
    np.random.shuffle(fileNames)

    def reader():
        batch_imgs = []
        batch_labels = []
        for name in fileNames:
            img = cv2.imread(dataDir+"Images/"+name[0])
            # 使用cv2将图像的色彩通道还原成RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform_img(img, augment=False)
            label = name[1]  # name[1]是文件名，name[2]是标签
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype(
                    'int64').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []
        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype(
                'int64').reshape(-1, 1)
            yield imgs_array, labels_array
    return reader


# 主函数
if __name__ == '__main__':
    x = np.random.randn(*[3, 3, 28, 28])
    x = x.astype('float32')
    train_loader = t_loader()
    data = next(train_loader())
    x, _ = data
