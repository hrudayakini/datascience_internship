import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/HRUDAYA TUSHAR KINI/Desktop/titanic_dataset/tested.csv")

#printing rows of the dataframe
print(df.head(10))

#getting information about the data
df.info()

#printing missing values

print(df.isnull().sum())

#handling the missing values
#drop the column cabin
df=df.drop(columns='Cabin', axis=1)

#replacing the missing values in "Age" column with value
df['Age'].fillna(df['Age'].mean(), inplace=True)

#finding the mode value of "Embarked" column
print(df['Embarked'].mode())

#only want s value not 0
print(df['Embarked'].mode()[0])
print(df)

#replacing the missing values in "Embarked" column with mode value
df['Embarked'].fillna(df['Embarked'].mode(), inplace=True)
#check the number of missing values in each column
print(df.isnull().sum())


#getting the statistical measure of the data
print(df.describe())


#finding the number of people who survived and not survived
print(df['Survived'].value_counts())

#data part
sns.set()#gives time for the plot

sns.countplot(data=df, x='Survived')
#0 indicates person who dies and 1 indicates who survived
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

#check count for male and female
print(df['Sex'].value_counts())

#plot graph of sex column
sns.countplot(data=df, x='Sex', color='green')
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

#for p class
sns.set()
sns.countplot(data=df, x='Pclass')
plt.show()

#to miss person from 1st class who survived and person from 2 class
sns.countplot(data=df, x='Pclass',hue='Survived')
plt.show()

sns.boxplot(df['Age'])
plt.show()

#encode categorical column
print(df['Sex'].value_counts())
print(df['Embarked'].value_counts())

# Select relevant features
features = ['Pclass', 'Sex', 'Age', 'Fare']

# Handle missing values
titanic_data = df[features + ['Survived']].dropna()

# Convert categorical variables into numerical representations
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})

# Split the data into training and testing sets
X = titanic_data[features]
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Train the classifier
rf_classifier.fit(X_train, y_train)
# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


