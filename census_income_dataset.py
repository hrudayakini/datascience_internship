import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("C:/Users/HRUDAYA TUSHAR KINI/Desktop/census_income_dataset/adult.csv")
print(df.head())

#getting the information
print(df.info())

print(df['income'].value_counts())
print(df['sex'].value_counts())
print(df['native.country'].value_counts())
print(df['workclass'].value_counts())
print(df['occupation'].value_counts())

#dropping

df=df.drop(['education', 'fnlwgt'], axis=1)
print(df.head(2))


#replacing with NAN

df.replace('?', np.NaN,inplace=True)

#fill missing values in 'workclass' column with mode
mode_workclass=df['workclass'].mode()[0]
df['workclass'].fillna(mode_workclass, inplace=True)

#fill missing values in 'occupation' column with mode
mode_occupation=df['occupation'].mode()[0]
df['occupation'].fillna(mode_occupation, inplace=True)

#fill missing values in 'native.country' column with mode
mode_native_country=df['native.country'].mode()[0]
df['native.country'].fillna(mode_native_country, inplace=True)


#check for missing values

print(df.isnull().sum())


#fill missing values in 'occupation' column with mode

print(df.head())

df.fillna(method= 'ffill',inplace=True)


#label encoding

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df['workclass']=le.fit_transform(df['workclass'])
df['marital.status']=le.fit_transform(df['marital.status'])
df['occupation']=le.fit_transform(df['occupation'])
df['relationship']=le.fit_transform(df['relationship'])
df['race']=le.fit_transform(df['race'])
df['sex']=le.fit_transform(df['sex'])
df['native.country']=le.fit_transform(df['native.country'])
df['income']=le.fit_transform(df['income'])

print(df.head())


#correlation
corrmat=df.corr()
print(corrmat)



#data analysis
sns.barplot(x='income', y='age', data=df)
plt.show()
#distribution of income
sns.countplot(x='income', data=df)
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Count')
plt.show()

# Creating a distribution plot for 'Age'
age = df['age'].value_counts()

plt.figure(figsize=(10, 5))
plt.style.use('fivethirtyeight')
sns.distplot(df['age'], bins=20)
plt.title('Distribution of Age', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.show()

# Creating a pie chart for 'Marital status'
marital = df['marital.status'].value_counts()

plt.style.use('default')
plt.figure(figsize=(10, 7))
plt.pie(marital.values, labels=marital.index, startangle=10, explode=(
    0, 0.20, 0, 0, 0, 0, 0), shadow=True, autopct='%1.1f%%')
plt.title('Marital distribution', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.legend()
plt.legend(prop={'size': 7})
plt.axis('equal')
plt.show()


#training
x=df.drop(['income'], axis=1)
y=df['income']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2)

from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
print(gb.fit(x_train,y_train))


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#printing matrics
y_pred=gb.predict(x_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred)*100)