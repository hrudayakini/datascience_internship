import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df= pd.read_csv("C:/Users/admin/Desktop/black friday/train.csv")

#print head
print(df.head(10))
#check shape of dataset
print(df.shape)

print(df.info())
#missing values
print(df.isnull().sum())

#visualization
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Gender', palette='mako')
plt.show()

#marital status
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='Gender', y='Marital_Status')
plt.show()

#purchase status
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='Gender', y='Purchase')
plt.show()

#occupation status
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='Occupation', y='Purchase') #occupation has direct impact on purchase
plt.show()

#comparing male and female genders with hue
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='Occupation', y='Purchase', hue='Gender')
plt.show()

#outlier detection
#checking presence of ouutlier
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Gender', y='Purchase')
plt.show()

#occupation outlier
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Occupation', y='Purchase')
plt.show()

#purchase outlier
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Age', y='Purchase')
plt.show()

#product category outlier
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Product_Category_1', y='Purchase')
plt.show()

#Data prepossing
print(df['Product_Category_2'].fillna(df['Product_Category_2'].mean(),inplace=True))
print(df['Product_Category_1'].mode())

#only want 5 value and not 0
print(df['Product_Category_1'].mode()[0])

df = df.drop('Product_Category_2', axis=1)
#encode categorical column
print(df['Product_ID'].value_counts())
print(df['Product_Category_1'].value_counts())

# Select relevant features
features = ['Product_ID', 'Occupation', 'Age', 'Marital_Status']
df.drop('Age', axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Product_ID'] = label_encoder.fit_transform(df['Product_ID'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['City_Category'] = label_encoder.fit_transform(df['City_Category'])
df['Stay_In_Current_City_Years'] = label_encoder.fit_transform(df['Stay_In_Current_City_Years'])

print(df)


#Correlation betn all the column and quality column
correlation=df.corr()

#Positive Correlation
#Negative Correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size': 8},cmap='Blues')
plt.show()

#separate the data and label
X= df.drop('Purchase',axis=1)
print(X)

#label Binarization
Y = df['Purchase'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(Y)

#Train & Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(Y.shape,Y_train.shape,Y_test.shape)

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
pipeline = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


