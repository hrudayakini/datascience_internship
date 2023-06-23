import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
test= pd.read_json("C:/Users/HRUDAYA TUSHAR KINI/Desktop/what's_cooking/test.json")
train= pd.read_json("C:/Users/HRUDAYA TUSHAR KINI/Desktop/what's_cooking/train.json")

print(test.head())
print(train.head())
print(train.shape)
print(train.describe())

print(train.info())
print(test.shape)
print(test.describe())

print(test.info())

print(test.isnull().sum())
print(train.isnull().sum())

#Finding count of unique cuisines in train dataframe
count_cuisine = train['cuisine'].value_counts(sort=True)
plt.figure(figsize=(10,10))
sns.barplot(y = count_cuisine.index, x = count_cuisine.values)
plt.title('Count of Unique Cuisines')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Cuisines', fontsize=12)
plt.show()

#Finding count of most common ingredients in train dataframe
count_ingredients = []
for x in train['ingredients']:
    for y in x:
        count_ingredients.append(y)
count_ingredients = pd.Series(count_ingredients)
c_ingredients = count_ingredients.value_counts(sort=True).head(10)
plt.figure(figsize=(10,10))
sns.barplot(y = c_ingredients.index, x = c_ingredients.values)
plt.title('Count of Most Common Ingredients')
plt.ylabel('Ingredients', fontsize=12)
plt.xlabel('Count', fontsize=12)
plt.show()


#Top 20 ingredients in all cuisines
cuisines = train['cuisine'].unique()
for i in cuisines:
  ingredients=[]
  for j in train[train['cuisine']==i]['ingredients']:
    for k in j:
      ingredients.append(k)
  ingredients = pd.Series(ingredients)
  c_ingredients = ingredients.value_counts(sort=True).head(20)
  plt.figure(figsize=(10,10))
  sns.barplot(y = c_ingredients.index, x = c_ingredients.values)
  plt.title(i)
  plt.ylabel('Ingredients', fontsize=12)
  plt.xlabel('Count', fontsize=12)
  plt.show()


print(x)
print(y)

cuisine_counts = train['cuisine'].value_counts()
print(cuisine_counts)
#
ax = cuisine_counts.plot(kind='bar',
                figsize=(12, 5),
                 title='Cuisine Distribution')
ax.set_xticks(range(len(cuisine_counts)))
ax.set_xlabel("Cuisine")
ax.set_xticklabels(cuisine_counts.index);
plt.show()


#
ingredients_count = dict()
for ingredients in train['ingredients']:
    for ingredient in ingredients:
        if ingredient in ingredients_count:
            ingredients_count[ingredient] = ingredients_count[ingredient] + 1
        else:
            ingredients_count[ingredient] = 1

ingredients_count = pd.DataFrame(ingredients_count.items())
ingredients_count = ingredients_count.sort_values(by=[1], ascending=False)
print(ingredients_count.shape)

#
ax = ingredients_count[:15].plot(kind='bar',
                figsize=(12, 5),
                title='Most used ingredients')
ax.set_xticks(range(len(ingredients_count[:15])))
ax.set_xlabel("ingredient")
ax.set_xticklabels(ingredients_count[0][:15]);
plt.show()

grouped_by_cuisines = train.groupby('cuisine')
print(grouped_by_cuisines.get_group('italian').head())

italian_ingredients = dict()
for ingredients in grouped_by_cuisines.get_group('italian').ingredients:
    for ingredient in ingredients:
        if ingredient in italian_ingredients:
            italian_ingredients[ingredient] = italian_ingredients[ingredient] + 1
        else:
            italian_ingredients[ingredient] = 1

italian_ingredients = pd.DataFrame(italian_ingredients.items())
italian_ingredients = italian_ingredients.sort_values(by=[1], ascending=False)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
vectorizer = TfidfVectorizer()

y = train.cuisine
X = train.ingredients.str.join(' ')
X = vectorizer.fit_transform(X)
X_test_data = test.ingredients.str.join(' ')
X_test_data = vectorizer.transform(X_test_data)
split = train_test_split(X, y)
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb_model = gnb.fit(x_train.toarray(), y_train)
pred = gnb_model.predict(x_test.toarray())
print("GaussianNB accuracy : ",accuracy_score(y_test, pred, normalize = True))


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

pred = neigh.predict(x_test)
# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(y_test, pred))

test_data_prediction = SVM_model.predict(X_test_data)
submission_df = pd.DataFrame(columns=['id', 'cuisine'])
submission_df['id'] = test ['id']
submission_df['cuisine'] = test_data_prediction
submission_df[['id' , 'cuisine' ]].to_csv("whats-cooking.csv", index=False)

