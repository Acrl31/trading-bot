import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Directory containing data files
DATA_DIR = "data"

# Parameters
TARGET_LOOKAHEAD = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

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
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
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
    
    # Create target variable
    df['future_price'] = df['close'].shift(-TARGET_LOOKAHEAD)
    df['target'] = (df['future_price'] > df['close']).astype(int)
    df = df.drop(columns=['future_price']).dropna()

    # Balance dataset
    df_up = df[df['target'] == 1]
    df_down = df[df['target'] == 0]
    df_down = resample(df_down, replace=True, n_samples=len(df_up), random_state=RANDOM_STATE)
    df_balanced = pd.concat([df_up, df_down])

    # Features and scaling
    feature_columns = [col for col in df.columns if col not in ['target', 'instrument']]
    X = df_balanced[feature_columns]
    y = df_balanced['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=feature_columns), y

def train_model(X, y):
    """
    Train and save a Gradient Boosting model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = GradientBoostingClassifier(
        n_estimators=200,  # Reduced for faster training
        learning_rate=0.05,
        max_depth=6,       # Reduced to prevent overfitting and speed up
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'trained_model.pkl')
    print("Model saved to 'trained_model.pkl'")
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