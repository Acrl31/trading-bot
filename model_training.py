import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Directory containing data files
DATA_DIR = "data"

# Parameters for training
TARGET_LOOKAHEAD = 5  # Number of steps ahead to predict
TEST_SIZE = 0.2       # Proportion of data for testing
RANDOM_STATE = 42     # Reproducibility

def load_data():
    """
    Load and combine historical data for all instruments.
    """
    all_data = []
    for file in os.listdir(DATA_DIR):
        if file.endswith("_data.csv"):
            file_path = os.path.join(DATA_DIR, file)
            df = pd.read_csv(file_path)
            df['instrument'] = file.split("_")[0]  # Add instrument column
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def preprocess_data(df):
    """
    Preprocess data: feature engineering, target creation, and scaling.
    """
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Feature engineering: Add moving averages, RSI, etc.
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=10).std()

    # Create target: Predict if price will go up or down
    df['future_price'] = df['close'].shift(-TARGET_LOOKAHEAD)
    df['target'] = (df['future_price'] > df['close']).astype(int)

    # Drop NaN rows (from feature and target creation)
    df = df.dropna()

    # Separate features and target
    features = ['open', 'high', 'low', 'close', 'volume', 'returns', 'volatility']
    X = df[features]
    y = df['target']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def train_model(X, y):
    """
    Train a Random Forest model on the processed data.
    """
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print("Preprocessing data...")
    X, y = preprocess_data(data)
    print("Training model...")
    trained_model = train_model(X, y)
    print("Model training complete.")