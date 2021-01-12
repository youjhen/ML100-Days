#https://aifreeblog.herokuapp.com/posts/64/Data_Analytics_in_Practice_Titanic/
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

data_path = 'C:/Users/tuser.LAPTOP-QMJF8JVH/Downloads/'
train = pd.read_csv(os.path.join(data_path, 'titanic_train.csv'))
test = pd.read_csv(os.path.join(data_path, 'titanic_test.csv'))
data=train.append(test)
#使用append合併之後會造成index重複問題，因此要將index重新設定
data.reset_index(inplace=True,drop=True)
data.describe()

import matplotlib.gridspec as gridspec
# Sex性別、Pclass票務艙、Embarked登船港口、SibSp兄弟姊妹配偶人數、Parch父母子女人數
selected_cols = ['Sex','Pclass','Embarked','SibSp','Parch']

plt.figure( figsize=(10,len(selected_cols)*5) )
gs = gridspec.GridSpec(len(selected_cols),1)    
for i, col in enumerate( data[selected_cols] ) :        
    ax = plt.subplot( gs[i] )
    sns.countplot( data[col], hue=data.Survived, palette=['lightcoral','skyblue'])
    ax.set_yticklabels([])
    ax.set_ylabel( 'Counts' )
    ax.legend( loc=1 )   # upper right:1 ; upper left:2
    for p in ax.patches:
        ax.annotate( '{:,}'.format(p.get_height()), (p.get_x(), p.get_height()+1.5) )
plt.show()
data[data.Survived].plot(kind="bar", stacked=True)
'''
透過上列的長條圖(Bar Plot)與統計表，我們可得到下列的觀察：
女性的生存率明顯高於男性；
票務艙(Pclass)的生存率大小: Pclass 1 > Pclass 2 > Pclasss 3；
登船港口(Embarked)的生存率以 C 最高；
當船上的兄弟姐妹配偶人數(SibSp)有 1 人同行時，則生存率較高；
當船上的父母子女人數(Parch)為 1~3 人時，有較高的生存率。
'''
#特徵工程(Feature Engineering)
# 定義用來統計欄位缺漏值總數的函數
def Missing_Counts( Data ) : 
    missing = Data.isnull().sum()  # 計算欄位中缺漏值的數量 
    missing = missing[ missing>0 ]
    missing.sort_values( inplace=True ) 
    
    Missing_Count = pd.DataFrame( { 'ColumnName':missing.index, 'MissingCount':missing.values } )  # Convert Series to DataFrame
    Missing_Count[ 'Percentage(%)' ] = Missing_Count['MissingCount'].apply( lambda x:round(x/Data.shape[0]*100,2) )
    return  Missing_Count
display( Missing_Counts(data) )
#票價(Fare) : 填補整體票價的中位數
data['Fare'].fillna( data.Fare.median(), inplace=True )
#登船港口(Embarked) : 計算 Embarked 欄位中每個相異值的次數,填補次數最多的港口 'S'
display( data['Embarked'].value_counts() )
df_data['Embarked'].fillna( 'S', inplace=True )
#船艙號碼(Cabin) :取出 Cabin 中的第一個字母，如果為缺漏值，則以 NoCabin 表示
data['Cabin'].apply( lambda x:str(x)[0] if not pd.isnull(x) else 'NoCabin' )
data['Cabin'].unique()

#獨熱編碼(OneHot Encoding)
OneHot_Embarked = pd.get_dummies( data.Embarked, prefix='Embarked' )
#合併 Embarked 編碼後的欄位
pd.concat([data, OneHot_Embarked], axis=1).drop( 'Embarked', axis=1, inplace=True )
#標籤編碼(Label Encoding)
Sex_mapping = { 'male':'0', 'female':'1' }
data.Sex.map(Sex_mapping)

# Method 1: split()
data['Title'] = data.Name.str.split(', ', expand=True)[1]
data['Title'] = data.Title.str.split('.', expand=True)[0]
a=data['Title'].unique()
# Method 2: 正規表示法(Regular Expression)
import re
regex = re.compile( ' ([A-Za-z]+)\.' )  
data['Title'] = data.Name.map( lambda x:regex.search(x)[0] )
# Dropping the first and the last words
data['Title'] = data.Title.map( lambda x:x[1:][:-1] )  
b=data['Title'].unique()
list(set(a).intersection(set(b)))
list(set(a).union(set(b)))
list(set(b).difference(set(a))) # b中有而a中沒有的
#上述擷取出的稱謂總歸為五類：Mr、Mrs、Miss、Master、Rare。
data['Title'] = data.Title.replace( ['Don','Rev','Dr','Major','Lady','Sir','Col','Capt','Countess','Jonkheer','Dona'], 'Rare' )
data['Title'] = data.Title.replace( ['Ms','Mlle'], 'Miss' )
data['Title'] = data.Title.replace( 'Mme', 'Mrs' )
data['Title'].unique()
'''
#pip install apyori
支持度計算方式 : SUPPORT (X ⇒ Y) = P(X , Y)
信賴度計算方式 : CONFIDENCE (X ⇒ Y) = P(Y | X)
正相關計算方式 : LIFT (X ⇒ Y) = P(Y | X) ／ P(Y) => LIFT值需大於 1
'''
from apyori import apriori
nominal_cols = ['Embarked','Pclass','Age', 'Survived', 'Sex']

train['Embarked'].fillna('Unknown',inplace=True)

rep = {0: "Dead", 1: "Survived"}
train.replace({'Survived' : rep}, inplace=True)

def binning(col, cut_points, labels=None):
  minval = col.min()
  maxval = col.max()
  break_points = [minval] + cut_points + [maxval]
  if not labels:
    labels = range(len(cut_points)+1)
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin
cut_points = [1, 10, 20, 50 ]
labels = ["Unknown", "Child", "Teen", "Adult", "Old"]
train['Age'] = binning(train['Age'], cut_points, labels)

in_titanic=train[nominal_cols]

dataset = []
for i in range(0, in_titanic.shape[0]-1):
    dataset.append([str(in_titanic.values[i,j]) for j in range(0, in_titanic.shape[1])])
# dataset = in_titanic.to_xarray()

rules=apriori(dataset, min_support=0.1, min_confidence=0.7)
results = list(rules)

df = pd.DataFrame(columns=('Items','Antecedent','Consequent','Support','Confidence','Lift'))

Support =[]
Confidence = []
Lift = []
Items = []
Antecedent = []
Consequent=[]

for RelationRecord in results:
    for ordered_stat in RelationRecord.ordered_statistics:
        Support.append(RelationRecord.support)
        Items.append(RelationRecord.items)
        Antecedent.append(ordered_stat.items_base)
        Consequent.append(ordered_stat.items_add)
        Confidence.append(ordered_stat.confidence)
        Lift.append(ordered_stat.lift)

df['Items'] = list(map(set, Items))                                   
df['Antecedent'] = list(map(set, Antecedent))#X：先導LHS
df['Consequent'] = list(map(set, Consequent))#Y：後繼RHS
df['Support'] = Support
df['Confidence'] = Confidence
df['Lift']= Lift

df.sort_values(by ='Confidence', ascending = False, inplace = True)
df

#線性回歸模型跟邏輯回歸模型
from sklearn.linear_model import LinearRegression, LogisticRegression
alg = LinearRegression()
x = train[['Pclass', 'Sex', 'Age', 'Embarked']]
y = train['Survived']
alg.fit(x, y)