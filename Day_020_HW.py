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

# 取出 EXT_SOURCE 的幾項變數並檢驗其相關性
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs

plt.figure(figsize = (8, 6))
# 繪製相關係數 (correlations) 的 Heatmap
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');

# 進一步我們檢查這三項變數在 Target 上的分布是否不同
plt.figure(figsize = (24, 8))

# 依不同 EXT_SOURCE 逐項繪製 KDE 圖形
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    # 做 subplot
    plt.subplot(1, 3, i + 1)
    
    # KDE 圖形
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    
    # 加上各式圖形標籤
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)

# 1.請用 numpy 建立一個 10 x 10, 數值分布自 -1.0 ~ 1.0 的矩陣並繪製 Heatmap
# print(np.random.randn(10,10)) # we want to limit range 0 ~1 or have a range... gaussian is a wide spread
print(np.random.random((10,10)))

matrix = (np.random.random((10,10))-0.5)*2
plt.figure(figsize=(10,10))
heatmap = sns.heatmap(matrix, vmin=0, vmax=0.8, cmap = plt.cm.RdYlBu_r, annot=True
                     ,mask=np.triu(np.ones_like(matrix,dtype=np.bool))#显示对脚线下面部分图
                     ,square=True, linewidths=.5#每个方格外框显示，外框宽度设置
                     ,cbar_kws={"shrink": .5})
plt.show()

#2.請用 numpy 建立一個 1000 x 3, 數值分布為 -1.0 ~ 1.0 的矩陣，並繪製 PairPlot (上半部為 scatter, 對角線為 hist, 下半部為 density)
nrow = 1000
ncol = 3
matrix = 2*(np.random.random((1000,3))-0.5)
indice = np.random.choice([0,1,2], size=nrow)
plot_data = pd.DataFrame(matrix, indice)
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False)
grid.map_upper(plt.scatter , alpha = 0.2)
grid.map_diag(plt.hist )
grid.map_lower(sns.kdeplot , cmap = plt.cm.OrRd_r)
plt.show()

#3.請用 numpy 建立一個 1000 x 3, 數值分布為常態分佈的矩陣，並繪製 PairPlot (上半部為 scatter, 對角線為 hist, 下半部為 density)
nrow = 1000
ncol = 3
matrix = np.random.randn(1000,3)
indice = np.random.choice([0,1,2], size=nrow)
plot_data = pd.DataFrame(matrix, indice)
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False)
grid.map_upper(plt.scatter , alpha = 0.2)
grid.map_diag(plt.hist )
grid.map_lower(sns.kdeplot , cmap = plt.cm.OrRd_r)

plt.show()