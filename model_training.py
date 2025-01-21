import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
import joblib

# Directory containing the data files
DATA_DIR = "data"
INSTRUMENTS = [
    'EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CHF', 'EUR_JPY', 'GBP_JPY', 'EUR_GBP', 'USD_CAD', 'NZD_USD'
]

# Function to preprocess the data
def preprocess_data(file_path):
    """
    Load and preprocess data for model training.
    """
    data = pd.read_csv(file_path)
    
    # Feature engineering with essential indicators for scalping
    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['Price_Change'] = data['close'].pct_change()  # Percent change in price
    data['Volatility'] = data['high'] - data['low']  # Intraday volatility
    data['Volume_Change'] = data['volume'].pct_change()  # Percent change in volume

    # Lag features to account for historical data
    data['Lag_Close_1'] = data['close'].shift(1)
    data['Lag_Volume_1'] = data['volume'].shift(1)

    # Handle missing values and prepare target variable
    data.dropna(inplace=True)
    data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, -1)

    # Features and labels
    X = data[['SMA_5', 'SMA_20', 'Price_Change', 'Volatility', 'Volume_Change', 'Lag_Close_1', 'Lag_Volume_1']]
    y = data['Target']
    
    return X, y

# Efficiently process data for multiple instruments
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
X_combined = pd.concat(X_combined, axis=0, ignore_index=True)
y_combined = pd.concat(y_combined, axis=0, ignore_index=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Use Random Forest (faster than Gradient Boosting for scalping)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Feature selection
selector = SelectFromModel(model, threshold="mean", max_features=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# TimeSeriesSplit for time-series cross-validation
tscv = TimeSeriesSplit(n_splits=3)
cross_val_scores = []

for train_idx, test_idx in tscv.split(X_train_selected):
    model.fit(X_train_selected[train_idx], y_train.iloc[train_idx])
    y_pred = model.predict(X_train_selected[test_idx])
    cross_val_scores.append(accuracy_score(y_train.iloc[test_idx], y_pred))

print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean score: {np.mean(cross_val_scores)}")

# Evaluate the model
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)
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

model_file_path = os.path.join(MODEL_DIR, "scalping_model.pkl")
joblib.dump(model, model_file_path)
print(f"Model saved to {model_file_path}")