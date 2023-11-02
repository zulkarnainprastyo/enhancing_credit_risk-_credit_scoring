# Import necessary libraries
import pandas as pd

# Load the dataset (replace 'LoanStats3c.csv' with the actual path to your dataset)
dataset = pd.read_csv('LoanStats3a.csv', skiprows=1, low_memory=False)

# Display the first few rows and columns
print(dataset.head())
print(dataset.columns)

def select_features(data, k=10, target_column='loan_status'):
    """
    Perform feature selection on the given dataset using traditional credit data.

    Parameters:
    data (DataFrame): The input dataset for feature selection.
    k (int): The number of top features to select.
    target_column (str): The name of the target column.

    Returns:
    selected_features (DataFrame): The dataset with selected features.
    """
    # Define a list of traditional credit data features from the Lending Club dataset
    traditional_credit_data_features = [
        'annual_inc', 'open_acc', 'total_acc', 'revol_util', 'dti'
    ]

    # Preprocess the 'revol_util' column to remove '%' and convert to float
    data['revol_util'] = data['revol_util'].str.rstrip('%').astype(float)

    # Separate the features and target variable
    X = data[traditional_credit_data_features]
    y = data[target_column]

    # Select the top k features based on F-statistic
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)

    # Create a DataFrame with the selected features
    selected_features = data.iloc[:, selected_indices]

    return selected_features
