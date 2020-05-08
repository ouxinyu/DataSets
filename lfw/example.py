'''
  @Author: Xinyu Ou
  @Date:   2020-05-07 11:04:20
  @Last Modified by:   Your name
  @Last Modified time: 2020-05-07 11:04:20

LFW (Labled Faces in the Wild)人脸数据集：是目前人脸识别的常用测试集，其中提供的人脸图片均来源于生活中的自然场景，因此识别难度会增大，尤其由于多姿态、光照、表情、年龄、遮挡等因素影响导致即使同一人的照片差别也很大。
并且有些照片中可能不止一个人脸出现，对这些多人脸图像仅选择中心坐标的人脸作为目标，其他区域的视为背景干扰。LFW数据集共有13233张人脸图像，每张图像均给出对应的人名，共有5749人，且绝大部分人仅有一张图片。每张图片的尺寸为250X250，绝大部分为彩色图像，但也存在少许黑白人脸图片。当然，在深度学习流行的今天，LFW数据集的识别率已经达到99.78%。
'''

# 此处介绍如何借助sklearn库的内置方法完成lfw数据集的读取使用方法，具体如下：

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

# 1. 运行以下代码，生成默认文件夹
# faces = fetch_lfw_people(min_faces_per_person=20, resize=0.8)

# 2. 将下载好的文件lfw-funneled.tgz，复制到sklearn的默认存储文件夹路径：
# > C: \Users\用户名\scikit_learn_data\fw_home

# 3. 使用以下语句载入和使用`lfw人脸识别数据集`
print('\r 数据载入中...', end='')
faces = fetch_lfw_people(min_faces_per_person=20,
                         resize=0.8)  # 利用sklearn内置方法读取LFW人脸识别库
image_shape = faces.images[0].shape   # 获取图像的尺寸
print('完成。')

# 显示照片
fig, axes = plt.subplots(2, 6, figsize=(12, 6), subplot_kw={
                         'xticks': (), 'yticks': ()})
for target, image, ax in zip(faces.target, faces.images, axes.ravel()):
    ax.imshow(image, cmap=plt.cm.gray)  # 灰度模式显示
    ax.set_title(faces.target_names[target])
plt.show()
