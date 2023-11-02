# deep_learning.py (perform classification)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
data = pd.read_csv('LoanStats3a.csv')

# Define the target variable
target_column = 'loan_status'  # Replace with your actual target column name

# Select relevant features (you can customize this)
selected_features = ['loan_amnt', 'int_rate', 'annual_inc', 'term', 'grade', 'installment']

# Check if the target column exists in the dataset
if target_column in data.columns:
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
else:
    print(f'Target column "{target_column}" not found in the dataset.')
