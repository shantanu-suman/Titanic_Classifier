# filepath: /titanic_model_project/titanic_model_project/src/TestModel.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import preprocess_data  # Assuming this function is defined in preprocessing.py

# Load the saved model
model = joblib.load(r'C:\Git\Titanic_Classifier\titanic_model.pkl')  # Adjust the path if necessary

# Load the new test data
test_df = pd.read_csv(r'C:\Git\Titanic_Classifier\test.csv')  # Replace with the actual path to your test CSV file

# Preprocess the test data
X_test = preprocess_data(test_df)

# Scale the features
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Make predictions
y_pred = model.predict(X_test)

# Output predictions
print("Predictions:", y_pred)