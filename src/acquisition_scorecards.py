# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset from your local directory (update the file path accordingly)
data = pd.read_csv('LoanStats3a.csv', low_memory=False)

# Data Preprocessing
# Include data cleaning, transformation, and feature engineering steps here

# Define your target variable and acquisition features
target_column = 'acquisition_target'  # Replace with your actual target variable
acquisition_features = ['acquisition_feature1', 'acquisition_feature2', 'acquisition_feature3']
# Add more features as needed for your acquisition scorecard

# Ensure that the specified columns exist in your acquisition dataset
if set(acquisition_features + [target_column]).issubset(data.columns):
    # Split the data into training and testing sets
    X_acquisition = data[acquisition_features]
    y_acquisition = data[target_column]
    X_acquisition_train, X_acquisition_test, y_acquisition_train, y_acquisition_test = train_test_split(X_acquisition, y_acquisition, test_size=0.2, random_state=42)

    # Model Development
    # Build a classification model (Random Forest Classifier) with hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    model_acquisition = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model_acquisition, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_acquisition_train, y_acquisition_train)

    # Get the best model from the grid search
    best_model_acquisition = grid_search.best_estimator_

    # Make predictions
    y_acquisition_pred = best_model_acquisition.predict(X_acquisition_test)

    # Evaluate the model
    accuracy_acquisition = accuracy_score(y_acquisition_test, y_acquisition_pred)
    classification_rep_acquisition = classification_report(y_acquisition_test, y_acquisition_pred)

    print("Acquisition Model Accuracy:", accuracy_acquisition)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Classification Report (Acquisition):\n", classification_rep_acquisition)

    # Save the best-trained acquisition model for future use
    joblib.dump(best_model_acquisition, 'acquisition_scorecard_model.pkl')

    # You can create scorecards, adjust decision thresholds, and further fine-tune your models as needed.

    # Finally, save the results, documentation, and reports as required for your project.

else:
    print("Specified columns do not exist in the acquisition dataset.")
