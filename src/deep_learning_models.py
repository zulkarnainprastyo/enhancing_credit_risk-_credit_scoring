# deep_learning_models.py
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# Load the dataset (replace 'LoanStats3c.csv' with your actual dataset)
data = pd.read_csv('LoanStats3a.csv')

# Define the target column (replace 'target_column' with your actual target column name)
target_column = 'loan_status'  # Replace with the actual target column name

# Define the list of selected features
selected_features = ['loan_amnt', 'int_rate', 'annual_inc', 'term', 'grade', 'installment']

# Check if all selected features and the target column exist in the dataset
if not all(col in data.columns for col in selected_features + [target_column]):
    print("Some specified columns do not exist in the dataset.")
else:
    # Filter the dataset to select relevant features and target
    data = data[selected_features + [target_column]]

    # Remove rows with missing target values
    data = data.dropna(subset=[target_column])

    # Preprocess the data
    label_encoder = LabelEncoder()
    data['term'] = label_encoder.fit_transform(data['term'])
    data['grade'] = label_encoder.fit_transform(data['grade'])

    # Define X (features) and y (target)
    X = data[selected_features]
    y = data[target_column]

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build a deep learning model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

    # Make predictions
    y_pred = model.predict_classes(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')