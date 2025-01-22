import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib  # To save the model

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
    df['volatility'] = df['returns'].rolling(window=10).std()
    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_long'] = df['close'].rolling(window=20).mean()
    df['ma_diff'] = df['ma_short'] - df['ma_long']
    df['ema_short'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_diff'] = df['ema_short'] - df['ema_long']
    rolling_std = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = df['ma_long'] + (2 * rolling_std)
    df['bollinger_lower'] = df['ma_long'] - (2 * rolling_std)
    df['bollinger_bandwidth'] = df['bollinger_upper'] - df['bollinger_lower']
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

    # Use Random Forest for potentially better performance
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Save the trained model to a file
    model_filename = "random_forest_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model

def cross_validate(X, y):
    """
    Perform cross-validation using a more powerful model and hyperparameter tuning.
    """
    # Using Random Forest for cross-validation
    model = RandomForestClassifier(
        n_estimators=50,  # Fewer trees for faster training
        max_depth=5,       # Shallower trees
        random_state=RANDOM_STATE
    )

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100],  # Use only one option for quicker tuning
        'max_depth': [10],      # Reduced number of hyperparameter options
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    grid_search = GridSearchCV(model, param_grid, cv=2, n_jobs=-1, verbose=2)  # Reduced to 2 splits
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    # Save the best model from cross-validation
    best_model_filename = "best_random_forest_model.pkl"
    joblib.dump(best_model, best_model_filename)
    print(f"Best model saved as {best_model_filename}")

    # Evaluate the model
    accuracies = []
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)  # Reduced to 2 splits
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)

    avg_accuracy = np.mean(accuracies)
    print(f"Cross-Validation Accuracy: {avg_accuracy:.2f}")
    return avg_accuracy


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print("Adding features...")
    data = add_features(data)
    print("Preprocessing data...")
    X, y = preprocess_data(data)
    
    # Train the full model and save it
    print("Training full model...")
    trained_model = train_model(X, y)

    # Perform cross-validation and save the best model
    print("Performing cross-validation...")
    cross_val_accuracy = cross_validate(X, y)
    
    print("Model training complete.")