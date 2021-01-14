#https://aifreeblog.herokuapp.com/posts/64/Data_Analytics_in_Practice_Titanic/
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件

# 忽略警告訊息
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
data['Embarked'].fillna( 'S', inplace=True )
#船艙號碼(Cabin) :取出 Cabin 中的第一個字母，如果為缺漏值，則以 NoCabin 表示
data['Cabin'].apply( lambda x:str(x)[0] if not pd.isnull(x) else 'NoCabin' )
data['Cabin'].unique()

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
TravelAlone=pd.DataFrame(np.where((train["SibSp"]+train["Parch"])>0, 0, 1))
#獨熱編碼(OneHot Encoding)
OneHot_Pclass= pd.get_dummies(train.Pclass, prefix='Pclass' )
#標籤編碼(Label Encoding)
Sex_mapping = { 'male':'0', 'female':'1' }
Label_Sex = train.Sex.map(Sex_mapping)

train['Age'].fillna(train['Age'].median(skipna=True), inplace=True)
train['Embarked'].fillna('S', inplace=True)
x = pd.get_dummies(train, columns=['Pclass','Embarked','Sex'])
cols = ['Age','Fare','Pclass_1','Pclass_2','Embarked_C','Embarked_S','Sex_male'] 
X = x[cols]
Y = train['Survived']

'''
Build a logreg and compute the feature importances
P(yi=0)=1-
'''
model = LogisticRegression(solver='liblinear', random_state=0).fit(X, Y)
model.intercept_
model.coef_
model.predict_proba(X)
model.predict(X)
model.score(X, Y)#(466+242)/891=0.7946127946127947
model

from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(Y, model.predict(X))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
from sklearn.feature_selection import RFE,RFECV
rfecv = RFECV(estimator=model, step=1, cv=10, scoring='accuracy').fit(X, Y)
# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

import statsmodels.api as sm

logit_model = sm.Logit(Y, X[['Embarked_C','Embarked_S']]).fit()
logit_model.summary()

linear_model =sm.OLS(Y, X[['Age','Fare']]).fit()
linear_model.summary()

'''
決策樹分類器
tree.DecisionTreeClassifier：分類樹
tree.DecisionTreeRegressor：回歸樹
tree.export_graphviz：將生成的決策樹導出為DOT格式，畫圖專用
tree.ExtraTreeClassifier：高隨機版本的分類樹
tree.ExtraTreeRegressor：高隨機版本的回歸樹
'''
train['Sex'].replace(['female','male'],[0,1], inplace=True)
train['Sex']=train['Sex'].astype('str')
train['Pclass']=train['Pclass'].astype('str')
train['Age'].fillna(train['Age'].median(skipna=True), inplace=True)
Features=train[['Pclass','Age','Sex']]
Ans=train['Survived']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(Features,Ans,test_size=0.1)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_predict=clf.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(clf.predict(x_test),y_test)
print(classification_report(y_test,y_predict,target_names=['died','survived']))

import graphviz
from sklearn.tree import export_graphviz
os.environ["PATH"] += os.pathsep + 'C:/Users/tuser.LAPTOP-QMJF8JVH/anaconda3/Lib/site-packages/graphviz/'
g=export_graphviz(clf, feature_names=['Pclass','Age','Sex'], class_names='survived', filled= True)
graphviz.Source(g)

'''
隨機森林參數：
n_estimators: 樹的數量(default=10)。
min_samples_leaf: 最終葉節點最少樣本數(default=1)；
                  當樣本不大時，可不設定使用預設，若樣本數量非常大時，則推薦增加此參數值。
min_samples_split:節點再劃分時所需的最小樣本數(default=2)；
                  當樣本不大時，可不設定使用預設，若樣本數量非常大時，則推薦增加此參數值。
oob_score: 是否採用袋外樣本(out-of-bag samples)來評估模型的準確度(default=False)。
'''
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier( n_estimators = 1000,
                              min_samples_split = 20,
                              min_samples_leaf = 1,
                              oob_score = True,
                              random_state = 1,
                              n_jobs = -1 )

rfc.fit(x_train, y_train)
rfc.oob_score_

from sklearn import svm,AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
# 支援向量機
svc = svm.SVC()
# 貝葉斯
nb = MultinomialNB()
# K鄰近
knn = KNeighborsClassifier()
# AdaBoost
boost = AdaBoostClassifier()
# xgboost
xgboost = XGBClassifier(objective='binary:logistic')
print('SVM acc is', np.mean(cross_val_score(svc, x_train, y_train, cv=10)))
print('NaiveBayes acc is', np.mean(cross_val_score(nb, x_train, y_train, cv=10)))
print('KNN acc is', np.mean(cross_val_score(knn, x_train, y_train, cv=10)))
print('AdaBoost acc is', np.mean(cross_val_score(boost, x_train, y_train, cv=10)))

print('xgboost acc is', np.mean(cross_val_score(xgboost, x_train, y_train, cv=10)))
# ValueError: DataFrame.dtypes for data must be int, float, bool or categorical.

from xgboost import plot_tree
import matplotlib.pyplot as plt

xgb = XGBClassifier()
xgb.fit(x_train, y_train)

plot_tree(xgb)
plt.show()