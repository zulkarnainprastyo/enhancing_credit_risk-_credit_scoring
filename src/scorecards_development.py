from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset from your local directory (update the file path accordingly)
data = pd.read_csv('LoanStats3a.csv', skiprows=1, low_memory=False)

# Data Preprocessing
# Convert the 'int_rate' column from string to float
data['int_rate'] = data['int_rate'].str.rstrip('%').astype('float') / 100.0

# Extract the numeric part of the 'term' column and convert it to an integer
data['term'] = data['term'].str.extract('(\d+)').astype(float).fillna(36).astype(int)  # Fill missing values with 36 months

# Check for NaN values in the dataset
nan_columns = data.columns[data.isna().any()].tolist()
print("Columns with NaN values:", nan_columns)

# Handle NaN values in numeric features
numeric_features = ['loan_amnt', 'int_rate', 'annual_inc', 'term', 'installment']
imputer = SimpleImputer(strategy='mean')
data[numeric_features] = imputer.fit_transform(data[numeric_features])

# One-hot encode the 'grade' column
data = pd.get_dummies(data, columns=['grade'], prefix='grade')

# Create a binary classification target variable: 'Fully Paid' vs. 'Charged Off'
data['loan_status'] = data['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

# Define your target variable and features
target_column = 'loan_status'
selected_features = ['loan_amnt', 'int_rate', 'annual_inc', 'term', 'grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G', 'installment']

# Split the data into training and testing sets
X = data[selected_features]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Development
# Build a classification model (Logistic Regression as an example)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Save the trained model for future use
joblib.dump(model, 'credit_scorecard_model.pkl')

# You can create scorecards, adjust decision thresholds, and further fine-tune your models as needed.

# Finally, save the results, documentation, and reports as required for your project.
