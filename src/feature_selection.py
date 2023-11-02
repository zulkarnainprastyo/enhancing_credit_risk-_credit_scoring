# Import necessary libraries
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Define a function for feature selection
def select_features(data, k=10):
    """
    Perform feature selection on the given dataset.

    Parameters:
    data (DataFrame): The input dataset for feature selection.
    k (int): The number of top features to select.

    Returns:
    selected_features (DataFrame): The dataset with selected features.
    """
    # Your feature selection code goes here
    # Example feature selection using ANOVA F-statistic:

    # Separate the features and target variable
    X = data.drop(columns=['loan_status'])
    y = data['loan_status']

    # Select the top k features based on F-statistic
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)

    # Create a DataFrame with the selected features
    selected_features = data.iloc[:, selected_indices]

    return selected_features

# Example: Load the dataset and perform feature selection
if __name__ == "__main":
    # Load the dataset (replace 'your_dataset.csv' with the actual path to your dataset)
    dataset = pd.read_csv('LoanStats3a.csv')

    # Perform feature selection
    k = 10  # Specify the number of top features to select
    selected_features_data = select_features(dataset, k)

    # Save the dataset with selected features to a new CSV file
    selected_features_data.to_csv('selected_features_dataset.csv', index=False)
