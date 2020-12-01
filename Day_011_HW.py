# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

# 設定 data_path
data_path = 'C:/Users/tuser.LAPTOP-QMJF8JVH/Downloads/'
f_app = os.path.join(data_path, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()

# 1: 計算 AMT_ANNUITY 的 q0 - q100
q_all = app_train['AMT_ANNUITY'].quantile(np.linspace(0,1,101))

q_pd=pd.DataFrame({'q': list(range(101)),
              'value': q_all})
q_pd
    
# 2.1 將 NAs 以 q50 填補
print("Before replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))
q_50 = list(q_all)[50]
app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50
print("After replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))
#Before replace NAs, numbers of row that AMT_ANNUITY is NAs: 12
#After replace NAs, numbers of row that AMT_ANNUITY is NAs: 0

# 2.2 Normalize values to -1 to 1
print("== Original data range ==")
print(app_train['AMT_ANNUITY'].describe())

# min max normalization：會將特徵數據按比例縮放到0到1的區間，（或是-1到1）
def normalize_value(x):    
    x=2*(((x-x.min())/(x.max()-x.min()))-0.5)
    return x

app_train['AMT_ANNUITY_min max normal'] = normalize_value(app_train['AMT_ANNUITY'])
print("== min max normal ==")
app_train['AMT_ANNUITY_min max normal'].describe()

# standard deviation normalization：會將所有特徵數據縮放成平均為0、平方差為1
from sklearn import preprocessing
app_train['AMT_ANNUITY_std normal'] = preprocessing.scale(app_train['AMT_ANNUITY'])
print("== standard deviation normal ==")
app_train['AMT_ANNUITY_std normal'].describe()

# 3
print("Before replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))

# 列出重複最多的數值
value_most = app_train['AMT_GOODS_PRICE'].value_counts().argmax()
mode_goods_price = list(app_train['AMT_GOODS_PRICE'].value_counts().index)
app_train.loc[app_train['AMT_GOODS_PRICE'].isnull(), 'AMT_GOODS_PRICE'] = mode_goods_price[0]
print("After replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))
# Before replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: 278
# mode good price zero is 450000.0
# After replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: 0