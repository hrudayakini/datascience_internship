import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/Users/admin/Desktop/boston_housing_dataset/Boston.csv")
print(df.head())
print(df.columns.values)

#statistical data
print(df.describe())
print(df.info())

print(df._data)

#drop
df = df.drop(columns='zn', axis=1)
print(df.drop)

#replace missing value in age column with mean value
df['age'].fillna(df['age'].mean(),inplace=True)
print(df['rad'].mode())

#only want 24 value and not 0
print(df['rad'].mode()[0])

#replace missing value in embarked column with mode value
print(df['rad'].fillna(df['rad'].mode(),inplace=True))
print(df.isnull().sum())


#statistical data
print(df.value_counts())
df = df.drop(columns='age', axis=1)
print(df.drop)


#plot graph
plt.figure(figsize=(12,6))
plt.hist(df['crim'], color='g');
plt.xlabel('Crime')
plt.ylabel('Frequency')
plt.show()

sns.countplot(data=df, x='tax')
plt.show()

sns.histplot(df['rad'], color='red')
plt.show()

sns.distplot(df['rad'], color='red')
plt.show()

sns.countplot(data=df,x='medv',palette='cubehelix')
plt.show()

sns.distplot(df['medv'])
plt.show()

print(df.corr())

sns.jointplot(x='rad', y='medv', data=df,kind='hex',color='g')
plt.show()

sns.jointplot(x='ptratio',y='medv',data=df,kind='reg',color='r')
plt.show()

sns.boxplot(data=df, x='medv')
plt.show()

#encode categorical column
print(df['age'].value_counts())
print(df['rad'].value_counts())

#training model
X=df.drop('medv', axis=1)
y= df['medv']
import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33,random_state=5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)

predictions= lin_reg.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Prices')
plt.ylabel('Predicted Prices')
plt.title('Prices vs Predicted prices')
plt.show()

lin_reg.score(X_test, y_test)
error= y_test-predictions
sns.distplot(error)
plt.show()
#accuracy
accuracy_score= sklearn.metrics.mean_squared_error(y_test, predictions)
print("Accuracy:",accuracy_score)
