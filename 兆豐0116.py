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
data.info()
data.describe()
'''
RangeIndex: 1309 entries, 0 to 1308
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  1309 non-null   int64  
 1   Survived     891 non-null    float64
 2   Pclass       1309 non-null   int64  
 3   Name         1309 non-null   object 
 4   Sex          1309 non-null   object 
 5   Age          1046 non-null   float64
 6   SibSp        1309 non-null   int64  
 7   Parch        1309 non-null   int64  
 8   Ticket       1309 non-null   object 
 9   Fare         1308 non-null   float64
 10  Cabin        295 non-null    object 
 11  Embarked     1307 non-null   object 
dtypes: float64(3), int64(4), object(5)

       PassengerId    Survived  ...        Parch         Fare
count  1309.000000  891.000000  ...  1309.000000  1308.000000
mean    655.000000    0.383838  ...     0.385027    33.295479
std     378.020061    0.486592  ...     0.865560    51.758668
min       1.000000    0.000000  ...     0.000000     0.000000
25%     328.000000    0.000000  ...     0.000000     7.895800
50%     655.000000    0.000000  ...     0.000000    14.454200
75%     982.000000    1.000000  ...     0.000000    31.275000
max    1309.000000    1.000000  ...     9.000000   512.329200
'''

Survived_Counts = data['Survived'].value_counts().reset_index()
Survived_Counts.columns = ['Survived','Counts']
Survived_Counts

plt.figure( figsize=(10,5) )
data['Survived'].value_counts().plot( kind='pie', colors=['lightcoral','skyblue'], autopct='%1.2f%%' )
plt.title( 'Survival' )  # 圖標題
plt.ylabel( 'ratio' )
plt.show()

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
'''
  ColumnName  MissingCount  Percentage(%)
0       Fare             1           0.08
1   Embarked             2           0.15
2        Age           263          20.09
3   Survived           418          31.93
4      Cabin          1014          77.46
'''
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
'''
                        Items         Antecedent  ... Confidence      Lift
14      {Survived, 1, female}        {1, female}  ...   0.968085  2.519286
16            {male, 2, Dead}          {2, Dead}  ...   0.938144  1.449563
15              {Adult, 2, S}         {Adult, 2}  ...   0.902439  1.247160
18               {male, 2, S}          {male, 2}  ...   0.898148  1.241230
1                      {2, S}                {2}  ...   0.891304  1.231772
..                        ...                ...  ...        ...       ...
34       {Adult, S, Survived}  {Adult, Survived}  ...   0.715054  0.988195
58  {3, Adult, S, male, Dead}   {3, Adult, Dead}  ...   0.710227  1.433338
10                {nan, Dead}              {nan}  ...   0.706215  1.146955
4                   {3, male}                {3}  ...   0.706122  1.091057
13                {male, nan}              {nan}  ...   0.700565  1.082470

[63 rows x 6 columns]
'''

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
x_test=pd.get_dummies(test, columns=['Pclass','Embarked','Sex']).tail()
X_test = x_test[cols]
'''
Build a logreg and compute the feature importances
p=(yi=1｜x)=>log(p/(1-p)=βtx+ε
勝算比odds=exp(βtx)=p/(1-p)
'''
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', random_state=0).fit(X , Y)
model.intercept_
model.coef_
model.predict_proba(X_test)
model.predict(X_test)
'''
array([[0.8983561 , 0.1016439 ],
       [0.61339562, 0.38660438],
       [0.8552473 , 0.1447527 ],
       [0.91130027, 0.08869973],
       [0.43518388, 0.56481612]])

array([0, 0, 0, 0, 1], dtype=int64)
预测X_test第一筆的Dead的概率为0.9，Survived的概率为0.1
'''
model.score(X, Y)#(466+242)/891=0.7946127946127947
from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(Y, model.predict(X))
'''
array([[466,  83],
       [100, 242]], dtype=int64)
'''
fig, ax = plt.subplots(figsize=(6, 6))
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
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=model, step=1, cv=10, scoring='accuracy').fit(X, Y)
# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

import statsmodels.api as sm

logit_model = sm.Logit(Y, X).fit()
logit_model.summary()
"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:               Survived   No. Observations:                  891
Model:                          Logit   Df Residuals:                      884
Method:                           MLE   Df Model:                            6
Date:                Fri, 15 Jan 2021   Pseudo R-squ.:                  0.3118
Time:                        18:48:39   Log-Likelihood:                -408.36
converged:                       True   LL-Null:                       -593.33
Covariance Type:            nonrobust   LLR p-value:                 8.048e-77
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Age           -0.0156      0.006     -2.572      0.010      -0.028      -0.004
Fare           0.0015      0.002      0.686      0.493      -0.003       0.006
Pclass_1       1.9533      0.272      7.178      0.000       1.420       2.487
Pclass_2       1.2466      0.227      5.493      0.000       0.802       1.691
Embarked_C     1.0205      0.287      3.551      0.000       0.457       1.584
Embarked_S     0.4837      0.222      2.178      0.029       0.048       0.919
Sex_male      -2.4499      0.187    -13.072      0.000      -2.817      -2.083
==============================================================================
"""
linear_model = sm.OLS(Y, X).fit()
linear_model.summary()
"""
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:               Survived   R-squared (uncentered):                   0.537
Model:                            OLS   Adj. R-squared (uncentered):              0.533
Method:                 Least Squares   F-statistic:                              146.3
Date:                Fri, 15 Jan 2021   Prob (F-statistic):                   5.27e-143
Time:                        18:48:55   Log-Likelihood:                         -494.91
No. Observations:                 891   AIC:                                      1004.
Df Residuals:                     884   BIC:                                      1037.
Df Model:                           7                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Age            0.0029      0.001      2.886      0.004       0.001       0.005
Fare           0.0008      0.000      2.206      0.028    8.73e-05       0.001
Pclass_1       0.2398      0.046      5.228      0.000       0.150       0.330
Pclass_2       0.1943      0.038      5.110      0.000       0.120       0.269
Embarked_C     0.5044      0.048     10.580      0.000       0.411       0.598
Embarked_S     0.4092      0.037     11.136      0.000       0.337       0.481
Sex_male      -0.4022      0.030    -13.398      0.000      -0.461      -0.343
==============================================================================
Omnibus:                       49.900   Durbin-Watson:                   1.872
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               53.293
Skew:                           0.570   Prob(JB):                     2.68e-12
Kurtosis:                       2.632   Cond. No.                         255.
==============================================================================

"""
from statsmodels.formula.api import glm as glm_sm
glm_model = glm_sm('Y ~ X', train, family=sm.families.Binomial()).fit()
glm_model.summary()
"""
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:                      Y   No. Observations:                  891
Model:                            GLM   Df Residuals:                      883
Model Family:                Binomial   Df Model:                            7
Link Function:                  logit   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -399.16
Date:                Fri, 15 Jan 2021   Deviance:                       798.33
Time:                        18:49:28   Pearson chi2:                     936.
No. Iterations:                     5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      1.5222      0.358      4.253      0.000       0.821       2.224
X[0]          -0.0328      0.007     -4.394      0.000      -0.047      -0.018
X[1]           0.0002      0.002      0.089      0.929      -0.004       0.004
X[2]           2.2803      0.289      7.901      0.000       1.715       2.846
X[3]           1.3304      0.234      5.685      0.000       0.872       1.789
X[4]           0.0074      0.370      0.020      0.984      -0.718       0.732
X[5]          -0.5302      0.318     -1.665      0.096      -1.154       0.094
X[6]          -2.5766      0.189    -13.633      0.000      -2.947      -2.206
==============================================================================
"""
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
clf.fit(x_train, y_train).feature_importances_
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
'''
交叉驗證Cross-Validation
 K-fold是拆解成 k 個子樣本來做交叉測試
'''
from sklearn.model_selection import cross_val_score

# 邏輯斯回歸：LR acc is 0.8239700374531835 變數不同：7 columns
acc_LR=np.mean(rfecv.grid_scores_)

# 隨機森林：RFC acc is 0.8239700374531835
acc_RFC=rfc.oob_score_

# 支援向量機：SVM acc is 0.6329783950617284
from sklearn import svm
svc = svm.SVC()
acc_SVM=np.mean(cross_val_score(svc, x_train, y_train, cv=10))

# 貝葉斯：NaiveBayes acc is 0.7141666666666666
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
acc_NB=np.mean(cross_val_score(nb, x_train, y_train, cv=10))

# K鄰近：KNN acc is 0.7790586419753086
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
acc_KNN=np.mean(cross_val_score(knn, x_train, y_train, cv=10))

# AdaBoost：AdaBoost acc is 0.8002623456790122
from sklearn.ensemble import AdaBoostClassifier
boost = AdaBoostClassifier()
acc_boost=np.mean(cross_val_score(boost, x_train, y_train, cv=10))

# xgboos：xgboost acc is
from xgboost.sklearn import XGBClassifier
xgboost = XGBClassifier()
acc_xgb=np.mean(cross_val_score(xgboost, x_train, y_train, cv=10))
'''
# ValueError: DataFrame.dtypes for data must be int, float, bool or categorical.
Int64Index: 801 entries, 798 to 202
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Pclass  801 non-null    object 
 1   Age     801 non-null    float64
 2   Sex     801 non-null    object 
dtypes: float64(1), object(2)
'''
# 梯度下降法Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
SGD = SGDClassifier()
acc_SGD=np.mean(cross_val_score(SGD, x_train, y_train, cv=10))

models = pd.DataFrame({
    'Model': ['LogisticRegression','Random Forest','Support Vector Machines',
              'Naive Bayes','KNN', 'AdaBoost', 'Stochastic Gradient Decent'],
    'Score': [acc_LR, acc_RFC, acc_SVM, acc_NB, acc_KNN, acc_boost, acc_SGD]})
models.sort_values(by='Score', ascending=False)
'''
                        Model     Score
1               Random Forest  0.801498
5                    AdaBoost  0.786528
0          LogisticRegression  0.784503
4                         KNN  0.750262
6  Stochastic Gradient Decent  0.740262
3                 Naive Bayes  0.699151
2     Support Vector Machines  0.628025
'''
