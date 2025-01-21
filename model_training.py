import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
from concurrent.futures import ProcessPoolExecutor

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
    
    # Feature engineering
    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['EMA_5'] = data['close'].ewm(span=5, adjust=False).mean()
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + (data['close'].diff().clip(lower=0).rolling(window=14).mean() / 
                                    -data['close'].diff().clip(upper=0).rolling(window=14).mean())))
    data['Price_Change'] = data['close'].pct_change()
    data['Volatility'] = data['high'] - data['low']
    data['Volume_Change'] = data['volume'].pct_change()
    data['Lag_Close_1'] = data['close'].shift(1)
    data['Lag_Volume_1'] = data['volume'].shift(1)

    # Drop missing values and set target
    data.dropna(inplace=True)
    data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, -1)

    # Features and labels
    X = data[['SMA_5', 'SMA_20', 'EMA_5', 'EMA_20', 'RSI', 'Price_Change', 'Volatility', 'Volume_Change', 'Lag_Close_1', 'Lag_Volume_1']]
    y = data['Target']
    
    return X, y

# Efficiently process data for multiple instruments
def process_instrument_data(instrument):
    file_path = os.path.join(DATA_DIR, f"{instrument}_data.csv")
    if os.path.exists(file_path):
        print(f"Processing data for {instrument}...")
        return preprocess_data(file_path)
    else:
        print(f"Data file for {instrument} not found. Skipping...")
        return None, None

# Process data in parallel
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_instrument_data, INSTRUMENTS))

# Combine all data into single datasets
X_combined = []
y_combined = []

for X, y in results:
    if X is not None and y is not None:
        X_combined.append(X)
        y_combined.append(y)

X_combined = pd.concat(X_combined, axis=0, ignore_index=True)
y_combined = pd.concat(y_combined, axis=0, ignore_index=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Use Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

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

model_file_path = os.path.join(MODEL_DIR, "scalping_model_rf.pkl")
joblib.dump(best_model, model_file_path)
print(f"Model saved to {model_file_path}")