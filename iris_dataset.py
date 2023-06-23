import pandas as pd
from sklearn.datasets import load_iris
irs=load_iris()
print(irs)
print(irs.keys())
print(irs.data)
print(irs.target)
print(irs.feature_names)
print(irs.target_names)
print(irs.DESCR)
df=pd.read_csv("C:/Users/admin/Desktop/hrudaya_datasets/IRIS.csv")
print(df)
print(df.head(10))
print(df.tail())
print(df.columns.values)
print(df.describe())

#preparing x and y

X = df.drop('species', axis=1)
Y = df['species']
print(X)
print(Y)



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures=SelectKBest(score_func=chi2, k='all')
fit=bestfeatures.fit(X,Y)
dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(X.columns)
featuresScores=pd.concat([dfcolumns,dfscores],axis=1)
featuresScores.columns=['Specs','Score']

print(featuresScores)



#Feature Engineering
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model= ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

feat_importance=pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(4).plot(kind='barh')
plt.show()

#numerical to categorial
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
df['sepal_length']=pd.cut(df['sepal_length'],3,labels=['0','1','2'])
df['sepal_width']=pd.cut(df['sepal_width'],3,labels=['0','1','2'])
df['petal_length']=pd.cut(df['petal_length'],3,labels=['0','1','2'])
df['petal_width']=pd.cut(df['petal_width'],3,labels=['0','1','2'])

print(df)


#DEALING WITH MISSING VALUES
'''1 Use DROP (fd.drop))
2. Use Replace (df.replace("back","dos"))
 3. fill NA()'''

#OVERSAMPLING AND UNDERSAMPLNG
from collections import  Counter
print(Counter(Y))
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
X,Y=ros.fit_resample(X,Y)
print(Counter(Y))

#identifying outliers by ploting
import pandas as pd
df=pd.read_csv("C:/Users/admin/Desktop/hrudaya_datasets/IRIS.csv")
from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['sepal_length'])
plt.show()


#dealing with outliers using interquantile range

print(df['sepal_length'])
Q1=df['sepal_length'].quantile(0.25)
Q3=df['sepal_length'].quantile(0.75)

IQR=Q3-Q1
print(IQR)

upper=Q1 +0.25*(IQR)
lower=Q3 -0.75*(IQR)

print(upper)
print(lower)

out1=df[df['sepal_length'] < lower].values
out2=df[df['sepal_length'] > upper].values


df['sepal_length'].replace(out1,lower,inplace=True)
df['sepal_length'].replace(out2,upper,inplace=True)

print(df['sepal_length'])



#PCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from  sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr=LogisticRegression()
pca=PCA(n_components=2)

X = df.drop('species', axis=1)
Y = df['species']

pca.fit(X)
X=pca.transform(X)

print(X)

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,random_state=0,test_size=0.3)
logr.fit(X_train,Y_train)

Y_pred=logr.predict(X_test)
print(accuracy_score(Y_test,Y_pred))



