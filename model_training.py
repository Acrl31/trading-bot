import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib

# Directory containing data files
DATA_DIR = "data"

# Parameters for training
TARGET_LOOKAHEAD = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 2  # Reduced folds for faster cross-validation
SUBSET_MODE = False  # Set True to use a smaller dataset for quick testing
SUBSET_SIZE = 5000  # Size of the subset when SUBSET_MODE is True

def load_data():
    """
    Load and combine historical data for all instruments.
    """
    all_data = []
    for file in os.listdir(DATA_DIR):
        if file.endswith("_data.csv"):
            file_path = os.path.join(DATA_DIR, file)
            df = pd.read_csv(file_path)
            df['instrument'] = file.split("_")[0]
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def add_features(df):
    """
    Add only the specified technical and statistical features to the data.
    """
    # Retaining only the features specified by the user
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

    # Optionally reduce dataset size for testing
    if SUBSET_MODE:
        df_balanced = df_balanced.sample(n=SUBSET_SIZE, random_state=RANDOM_STATE)

    # Standardize features (using only the selected features)
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long', 
                       'ema_short', 'ema_long', 'bollinger_upper', 'bollinger_lower', 
                       'rsi', 'macd', 'macd_signal', 'macd_diff', 'high_low_diff', 'open_close_diff']
    
    X = df_balanced[feature_columns]
    y = df_balanced['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def cross_validate_model(X, y):
    """
    Perform cross-validation to evaluate model performance.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,  # Reduced complexity for faster cross-validation
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        val_accuracy = model.score(X_val, y_val)
        cv_scores.append(val_accuracy)
    return np.mean(cv_scores), np.std(cv_scores)

def train_and_save_model(X, y):
    """
    Train the final model and save it to a file.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "trained_model.pkl")
    print("Model saved as trained_model.pkl")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {test_accuracy:.2f}")
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
    print("Cross-validating model...")
    cv_mean, cv_std = cross_validate_model(X, y)
    print(f"Cross-Validation Accuracy: {cv_mean:.2f} ± {cv_std:.2f}")
    print("Training and saving final model...")
    train_and_save_model(X, y)
    print("Model training complete.")