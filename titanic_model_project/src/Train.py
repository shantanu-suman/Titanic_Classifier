# filepath: /titanic_model_project/titanic_model_project/src/Train.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv(r'C:\Git\Titanic_Classifier\train.csv')
print(df.head())

# EDA 
print(df.info())
print(df.describe())
df.isnull().sum()
sns.heatmap(df.isnull(), cbar=False)
sns.countplot(x='Survived', data=df)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Handling missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(['Cabin'], axis=1, inplace=True)

# Feature Engineering
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(clf, 'titanic_model.pkl')