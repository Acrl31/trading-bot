import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Directory containing the data files
DATA_DIR = "data"
INSTRUMENTS = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "XAU_USD", "XAG_USD"]

# Function to preprocess the data
def preprocess_data(file_path):
    """
    Load and preprocess data for model training.
    """
    data = pd.read_csv(file_path)
    
    # Feature engineering using all available data
    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['Price_Change'] = data['close'].pct_change()  # Percent change in price
    data['Volatility'] = data['high'] - data['low']  # Intraday volatility
    data['Volume_Change'] = data['volume'].pct_change()  # Percent change in volume

    # Lag features to account for historical data
    data['Lag_Close_1'] = data['close'].shift(1)
    data['Lag_Close_2'] = data['close'].shift(2)
    data['Lag_Volume_1'] = data['volume'].shift(1)

    # Time-based features (e.g., day of the week)
    data['Day_Of_Week'] = data['timestamp'].apply(lambda x: pd.to_datetime(x).dayofweek)
    data['Hour_Of_Day'] = data['timestamp'].apply(lambda x: pd.to_datetime(x).hour)

    # Lag features for time-based elements
    data['Lag_Hour_1'] = data['Hour_Of_Day'].shift(1)

    # Drop rows with NaN values created by rolling and lag features
    data.dropna(inplace=True)

    # Define the target: 1 for Buy, -1 for Sell
    data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, -1)

    # Features and labels
    X = data[['SMA_5', 'SMA_20', 'Price_Change', 'Volatility', 
              'Volume_Change', 'Lag_Close_1', 'Lag_Close_2', 'Lag_Volume_1',
              'Day_Of_Week', 'Hour_Of_Day', 'Lag_Hour_1']]
    y = data['Target']
    
    return X, y

# Efficiently process data for multiple instruments using parallelism
from concurrent.futures import ProcessPoolExecutor

def process_instrument_data(instrument):
    file_path = os.path.join(DATA_DIR, f"{instrument}_data.csv")
    if os.path.exists(file_path):
        print(f"Processing data for {instrument}...")
        return preprocess_data(file_path)
    else:
        print(f"Data file for {instrument} not found. Skipping...")
        return None, None

# Using ProcessPoolExecutor for parallel processing of data
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_instrument_data, INSTRUMENTS))

# Collect the processed data
X_combined = []
y_combined = []

for X, y in results:
    if X is not None and y is not None:
        X_combined.append(X)
        y_combined.append(y)

# Combine all data into single datasets
X_combined = pd.concat(X_combined, axis=0, ignore_index=True)  # Concatenate along rows (axis=0)
y_combined = pd.concat(y_combined, axis=0, ignore_index=True)  # Concatenate along rows (axis=0)

# Check data shapes
print(f"X_combined shape: {X_combined.shape}")
print(f"y_combined shape: {y_combined.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Gradient Boosting model
model = GradientBoostingClassifier(random_state=42)

# Hyperparameter grid for Gradient Boosting
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
}

# Grid search for best hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Train with best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Cross-validation to evaluate model performance
cross_val_scores = cross_val_score(best_model, X_combined, y_combined, cv=3, n_jobs=-1)
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean score: {cross_val_scores.mean()}")

# Evaluate the model
y_pred = best_model.predict(X_test)
print("Model Performance:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')}")

# Save the trained model
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model_file_path = os.path.join(MODEL_DIR, "trading_model.pkl")
joblib.dump(best_model, model_file_path)
print(f"Model saved to {model_file_path}")