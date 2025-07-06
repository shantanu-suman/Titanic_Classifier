import joblib
import pandas as pd
from Train import clf, X_train, y_train  # Assuming clf is the trained model from Train.py

# Save the trained model to a file
joblib.dump(clf, 'titanic_model.pkl')

# Optionally, save the training data for reference
train_data = pd.DataFrame(X_train, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize'])
train_data['Survived'] = y_train
train_data.to_csv('titanic_train_data.csv', index=False)