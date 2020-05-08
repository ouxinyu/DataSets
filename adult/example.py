'''
  @Author: Xinyu Ou
  @Date:   2020-05-07 11:04:20
  @Last Modified by:   Your name
  @Last Modified time: 2020-05-07 11:04:20
'''
# adult.names 数据集的基本信息，文本文档（可改名）
# adult.data 训练集数据，下载后改名为：adult_train.csv
# adult.test 测试集数据，下载后改名为：adult_test.csv


import pandas as pd
training = pd.read_csv('adult/adult_train.csv', header=None, index_col=False,
                       names=['年龄', '单位性质', '权重', '学历', '受教育时长',
                              '婚姻状况', '职业', '家庭情况', '种族', '性别',
                              '资产所得', '资产损失', '周工作时长', '原籍',
                              '收入'])
testing = pd.read_csv('adult/adult_test.csv', header=None, index_col=False,
                      names=['年龄', '单位性质', '权重', '学历', '受教育时长',
                             '婚姻状况', '职业', '家庭情况', '种族', '性别',
                             '资产所得', '资产损失', '周工作时长', '原籍',
                             '收入'])

# 使用display()函数显示部分数据，以供预览
# .head()方法实现只显示前 5 行
# display(data.head())

# 为了方便展示，可以选取其中一部分特征（9个）进行显示
training_lite = training[['年龄', '单位性质', '学历',
                          '婚姻状况', '种族', '性别', '周工作时长', '职业', '收入']]
testing_lite = training[['年龄', '单位性质', '学历',
                         '婚姻状况', '种族', '性别', '周工作时长', '职业', '收入']]
print('训练集样例：\n{}'.format(training_lite.head()))
print('测试集样例：\n{}'.format(testing_lite.head()))
