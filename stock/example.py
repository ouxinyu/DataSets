'''
  @Author: Xinyu Ou
  @Date:   2020-05-07 11:04:20
  @Last Modified by:   Your name
  @Last Modified time: 2020-05-07 11:04:20

  该数据集是2020年2月15日收盘之后从证券交易软件中导出的数据。包含3776个样本，每个样本包含10种特征。可以用来演示各种回归算法，预测股票的涨跌。
'''
import pandas as pd
stocks = pd.read_excel('stock/stock.xls')

X = stocks.loc[:, '幅度%':'昨收']
y = stocks['涨跌']

print(stocks)
print('数据形态为：{}'.format(X.shape))
