import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
import requests
from zipfile import ZipFile
from io import BytesIO

# Download the dataset
url = 'https://resources.lendingclub.com/LoanStats3a.csv.zip'
response = requests.get(url)
with ZipFile(BytesIO(response.content)) as archive:
    with archive.open('LoanStats3a.csv') as file:
        data = pd.read_csv(file, low_memory=False)

# Data Preprocessing
# Clean and preprocess the data
# Feature selection, data cleaning, and transformation

# Define your target variable and features
target_column = 'loan_status'  # Replace with the correct target variable
selected_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util', 'total_pymnt', 'open_acc']

# Ensure the specified columns exist in your dataset
if set(selected_features + [target_column]).issubset(data.columns):
    # Handle missing values, if any
    data = data.dropna(subset=[target_column] + selected_features)

    # Split the data into training and testing sets
    X = data[selected_features]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Development
    # Build a classification model (Random Forest Classifier as an example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Create an imbalanced data pipeline
    oversampling = SMOTE(sampling_strategy=0.5, random_state=42)
    undersampling = EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=5)
    combine = SMOTEENN(sampling_strategy='majority')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('over', oversampling),
        ('under', undersampling),
        ('model', model)
    ])

    # Train the model using the imbalanced data pipeline
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = classification_report(y_test, y_pred)

    print("Classification Report:\n", accuracy)

    # Minimize false positives using the imbalanced data pipeline
    # You can further fine-tune the pipeline and use different techniques to minimize false positives.

    # Save the trained model for future use
    joblib.dump(model, 'credit_scorecard_model.pkl')

else:
    print("Specified columns do not exist in the dataset.")
