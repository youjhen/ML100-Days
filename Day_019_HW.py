# 載入需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件

# 忽略警告訊息
%matplotlib inline
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

# 設定 data_path
data_path = 'C:/Users/tuser.LAPTOP-QMJF8JVH/Downloads/'
f_app = os.path.join(data_path, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()

# 根據不同的 HOUSETYPE_MODE 對 AMT_CREDIT 繪製 Histogram
#To handle na items
app_train['HOUSETYPE_MODE'].fillna('na',inplace = True)
unique_house_type = app_train['HOUSETYPE_MODE'].unique()
nrows = len(unique_house_type)
ncols = nrows // 2

plt.figure(figsize=(10,30))
for i in range(len(unique_house_type)):  
    plt.subplot(nrows, ncols, i+1) 
    app_train.loc[(app_train['HOUSETYPE_MODE']==unique_house_type[i] ),'AMT_CREDIT' ].hist()    
    plt.title(str(unique_house_type[i]))
plt.show()