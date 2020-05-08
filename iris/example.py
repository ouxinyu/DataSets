'''
  @Author: Xinyu Ou
  @Date:   2020-05-07 11:04:20
  @Last Modified by:   Your name
  @Last Modified time: 2020-05-07 11:04:20
'''
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_datasets(show_examples=False):
    # 使用pandas载入数据集
    data = pd.read_csv('iris/iris.csv')

    # 将类别名称转换成编码
    data.loc[data.Species == 'Iris-setosa', 'Species'] = '0'
    data.loc[data.Species == 'Iris-versicolour', 'Species'] = '1'
    data.loc[data.Species == 'Iris-virginica', 'Species'] = '2'

    if show_examples is True:
        # 观察数据集，取前5个样本
        print(data.head())

    # 将数据中的特征和标签进行分离，其中第0位位索引号，第1-8位位特征，第9位为标签
    X = data.iloc[:, 0:4]
    y = data.iloc[:, 4]

    # 以 20%:80%的比例对训练集和测试集进行拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=1)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = prepare_datasets(show_examples=True)
