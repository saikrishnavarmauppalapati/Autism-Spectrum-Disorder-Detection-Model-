# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import joblib

# Load your dataset
# Replace the path with the actual file location if necessary
data = pd.read_csv(r'C:\Users\varma\OneDrive\Desktop\Toddler Autism dataset July 2018.csv')

# Split data into features (X) and target labels (y)
X = data.drop('autism', axis=1)  # Replace 'autism' with your target column name if it's different
y = data['autism']

# Data preprocessing
# Handle missing values and encode categorical variables
X.fillna(X.mean(), inplace=True)  # Fill missing values with mean
X = pd.get_dummies(X)  # One-hot encode categorical variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional but can help with model performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train a Logistic Regression model
model = LogisticRegression(max_iter=200)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display the results
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Calculate ROC AUC scores
print('Training ROC AUC:', roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
print('Validation ROC AUC:', roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Save the model and scaler for future use
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Function to get user input for a specific feature
def get_user_input(feature_name):
    while True:
        try:
            value = float(input(f"Enter {feature_name}: "))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid numeric value.")

# Load the saved model and StandardScaler
loaded_model = joblib.load('logistic_regression_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Prompt the user for input
print("\nPlease enter the following information:")
value_A1 = get_user_input("A1")
value_A2 = get_user_input("A2")
value_A3 = get_user_input("A3")
value_A4 = get_user_input("A4")
value_A5 = get_user_input("A5")
value_A6 = get_user_input("A6")
value_A7 = get_user_input("A7")
value_A8 = get_user_input("A8")
value_A9 = get_user_input("A9")
value_A10 = get_user_input("A10")
value_age = get_user_input("age")
value_Qchat_10_Score = get_user_input("Qchat-10-Score")
value_gender = input("Gender (Male/Female): ")

# Map gender to numerical values (e.g., Male=0, Female=1)
gender_mapping = {"Male": 0, "Female": 1}
value_gender = gender_mapping.get(value_gender.capitalize(), -1)  # Default to -1 if gender is not recognized

# Prepare input data as a DataFrame
new_data = pd.DataFrame({
    'A1': [value_A1],
    'A2': [value_A2],
    'A3': [value_A3],
    'A4': [value_A4],
    'A5': [value_A5],
    'A6': [value_A6],
    'A7': [value_A7],
    'A8': [value_A8],
    'A9': [value_A9],
    'A10': [value_A10],
    'age': [value_age],
    'Qchat-10-Score': [value_Qchat_10_Score],
    'gender': [value_gender]
})

# Standardize the new input data using the loaded scaler
new_data_scaled = loaded_scaler.transform(new_data)

# Make predictions using the loaded model
prediction = loaded_model.predict(new_data_scaled)

# Interpret the model's prediction
if prediction[0] == 1:  # Assuming 1 represents ASD
    print("The person is predicted to have ASD.")
else:
    print("The person is predicted not to have ASD.")
