import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt

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

def add_features(df):
    """
    Add technical and statistical features to the data.
    """
    df['returns'] = df['close'].pct_change()
    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_long'] = df['close'].rolling(window=20).mean()
    df['ema_short'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_diff'] = df['ema_short'] - df['ema_long']
    df['macd'] = df['ema_short'] - df['ema_long']  # Removed original 'macd' related columns
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()  # Removed the 'macd_diff' feature
    df['high_low_diff'] = df['high'] - df['low']
    df['open_close_diff'] = df['open'] - df['close']
    return df.dropna()

def preprocess_data(df):
    """
    Preprocess data: target creation, scaling, and balancing.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df['future_price'] = df['close'].shift(-TARGET_LOOKAHEAD)
    df['target'] = (df['future_price'] > df['close']).astype(int)
    df = df.drop(columns=['future_price']).dropna()

    # Balance the dataset
    df_up = df[df['target'] == 1]
    df_down = df[df['target'] == 0]
    df_balanced = pd.concat([
        resample(df_up, replace=True, n_samples=len(df_down), random_state=RANDOM_STATE),
        df_down
    ])

    # Standardize features
    feature_columns = [col for col in df.columns if col not in ['target', 'instrument']]
    X = df_balanced[feature_columns]
    y = df_balanced['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def train_model(X, y):
    """
    Train a Random Forest model and evaluate its performance.
    """
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
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
    print("Adding features...")
    data = add_features(data)
    print("Preprocessing data...")
    X, y = preprocess_data(data)
    print("Training model...")
    trained_model = train_model(X, y)
    print("Model training complete.")