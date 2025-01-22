import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
import joblib

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
    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_long'] = df['close'].rolling(window=20).mean()
    df['ema_short'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=20, adjust=False).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = df['ma_long'] + (2 * rolling_std)
    df['bollinger_lower'] = df['ma_long'] - (2 * rolling_std)
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['high_low_diff'] = df['high'] - df['low']
    df['open_close_diff'] = df['open'] - df['close']
    return df.dropna()

def preprocess_data(df):
    """
    Preprocess data: target creation, scaling, and balancing.
    """
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Create the target variable
    df['future_price'] = df['close'].shift(-TARGET_LOOKAHEAD)
    df['target'] = (df['future_price'] > df['close']).astype(int)
    df = df.drop(columns=['future_price']).dropna()

    # Select features and scale them
    feature_columns = [col for col in df.columns if col not in ['target', 'instrument']]
    X = df[feature_columns]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to DataFrame to retain feature names
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

    return X_scaled_df, y, feature_columns

def train_model(X, y):
    """
    Train a balanced Gradient Boosting model and ensure class balance.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Train the model
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        subsample=0.8  # Prevent overfitting
    )

    # Fit with adjusted sample weights
    sample_weights = [class_weight_dict[c] for c in y_train]
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Print metrics
    print("Classification Report (Balanced):")
    print(classification_report(y_test, y_pred, digits=4))

    # Analyze class distribution
    _, counts = np.unique(y_pred, return_counts=True)
    print(f"Predicted Class Distribution: {counts}")

    # Save the model
    joblib.dump(model, 'balanced_model.pkl')
    print("Balanced model saved to 'balanced_model.pkl'")

    return model

def balanced_predictions(model, X, threshold=0.5):
    """
    Make balanced predictions by adjusting the decision threshold.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print("Adding features...")
    data = add_features(data)
    print("Preprocessing data...")
    X, y, features = preprocess_data(data)
    print("Using features:", features)
    print("Training model...")
    trained_model = train_model(X, y)