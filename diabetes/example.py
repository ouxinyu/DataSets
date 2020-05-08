'''
  @Author: Xinyu Ou
  @Date:   2020-05-07 11:04:20
  @Last Modified by:   Your name
  @Last Modified time: 2020-05-07 11:04:20

该数据集包含数据集中共包含768个样本(entries)，每个样本8种特征。其中Outcome是样本的标签（即类别），0表示没有糖尿病，1表示患有糖尿病。此处的8种特征分别是：
- Pregnancies: 怀孕次数
- Glucose：血浆葡萄糖浓度
- BloodPressure：舒张压
- SkinThickness：肱三头肌皮肤褶皱厚度
- Insulin：两小时胰岛素含量
- BMI：身体质量指数，即体重除以身高的平方
- DiabetesPedigreeFunction：糖尿病血统指数，即家族遗传指数
- Age：年龄
'''
# 加载 pandas库，并使用read_csv()函数读取糖尿病预测数据集diabetes
import pandas as pd
data = pd.read_csv('diabetes/diabetes.csv')

# 输出数据集的形状，即展示数据包含的样本数(行)和特征数(列)
print("数据的形状为{}\n".format(data.shape))
data.info()
