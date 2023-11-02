import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Download the dataset from the provided URL
url = 'https://resources.lendingclub.com/LoanStats3a.csv.zip'
data = pd.read_csv(url, compression='zip', low_memory=False)

# Data Preprocessing
# Include data cleaning, transformation, and feature engineering steps here
# Replace these with actual feature columns and target variable
feature_columns = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']
target_column = 'loan_status'

# Check if the specified columns exist in the dataset
if set(feature_columns + [target_column]).issubset(data.columns):
    # Prepare the data for modeling
    X = data[feature_columns]
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Development
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model using the default threshold of 0.5
    print("Default Threshold (0.5)")
    classification_rep = classification_report(y_test, y_pred)
    print(classification_rep)

    # Set custom decision thresholds
    custom_threshold = 0.7  # Adjust this threshold as needed
    y_pred_custom = (model.predict_proba(X_test)[:, 1] > custom_threshold).astype(int)

    # Evaluate the model with the custom threshold
    print(f"Custom Threshold ({custom_threshold})")
    classification_rep_custom = classification_report(y_test, y_pred_custom)
    print(classification_rep_custom)

    # Save the trained model for future use
    joblib.dump(model, 'credit_scorecard_model.pkl')

    # You can fine-tune your decision thresholds and create scorecards as needed.

else:
    print("Specified columns do not exist in the dataset.")
