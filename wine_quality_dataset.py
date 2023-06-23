import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/admin/Desktop/wine_quality/winequality.csv")

print(df.shape)
#first Data
print(df.head())
#checking for missing values
print(df.isnull().sum())

#Data Analysis and Visulization
print(df.describe())  #stasticical measure of dataset

#drop
x = df.drop('quality',axis=1)
y = df['quality']
print(x)

x['fixed acidity'].fillna((df['fixed acidity'].mean()), inplace=True)
x['volatile acidity'].fillna((df['volatile acidity'].mean()), inplace=True)
x['citric acid'].fillna((df['citric acid'].mean()), inplace=True)
x['residual sugar'].fillna((df['residual sugar'].mean()), inplace=True)
x['chlorides'].fillna((df['chlorides'].mean()), inplace=True)
x['pH'].fillna((df['pH'].mean()), inplace=True)
x['sulphates'].fillna((df['sulphates'].mean()), inplace=True)

print(x.isnull().sum())

print(y.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x['type']=le.fit_transform(x['type'])
print(x)


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
bestFeatures=SelectKBest(score_func=f_regression,k='all')
fit=bestFeatures.fit(x,y)
dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x.columns)
featureScores=pd.concat((dfcolumns,dfscores),axis=1)
print(featureScores)

from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_)

#feature engineering
feat_importance = pd.Series(model.feature_importances_,index=x.columns)
feat_importance.nlargest(13).plot(kind='barh')
plt.show()

#number of values for each equality
sns.countplot(x='quality',data=df)
plt.ylabel("Count")
plt.show()


#volatile acidity vs Quality
ploat= plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data=df)
plt.show()

#citric acid vs Quality
ploat= plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data=df)
plt.show()

print(x.corr())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
lrmodel=LinearRegression()
dtmodel=DecisionTreeClassifier()
knn=KNeighborsClassifier()
gbr=GradientBoostingRegressor()
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
x_train , x_test ,y_train , y_test=train_test_split(x , y, random_state=5,test_size=0.1)
lrmodel.fit(x_train,y_train)

prediction1=lrmodel.predict(x_test)
print(r2_score(y_test,prediction1))

dtmodel.fit(x_train,y_train)

prediction2=dtmodel.predict(x_test)
print(accuracy_score(y_test,prediction2))
knn.fit(x_train,y_train)

prediction3=knn.predict(x_test)
print(accuracy_score(y_test , prediction3))

from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier()
rc.fit(x_train,y_train)

prediction4=rc.predict(x_test)
print(accuracy_score(y_test,prediction4))

user_input=np.array([[1 ,7.0,0.270,0.36,20.7,0.045,45.0,170.0,1.00100,3.00,0.450000,8.8]])
prediction=rc.predict(user_input)
print(prediction)



