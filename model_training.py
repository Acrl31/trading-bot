import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
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
    
    # Feature engineering with essential indicators for scalping
    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['EMA_5'] = data['close'].ewm(span=5, adjust=False).mean()
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + (data['close'].diff().clip(lower=0).rolling(window=14).mean() / 
                                    -data['close'].diff().clip(upper=0).rolling(window=14).mean())))
    data['Price_Change'] = data['close'].pct_change()  # Percent change in price
    data['Volatility'] = data['high'] - data['low']  # Intraday volatility
    data['Volume_Change'] = data['volume'].pct_change()  # Percent change in volume
    data['Lag_Close_1'] = data['close'].shift(1)
    data['Lag_Volume_1'] = data['volume'].shift(1)

    # Handle missing values and prepare target variable
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

# Wrapper for XGBClassifier to ensure compatibility with Scikit-learn
class XGBClassifierWrapper(xgb.XGBClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._estimator_type = "classifier"  # Explicitly set the type

xgb_model = XGBClassifierWrapper(
    use_label_encoder=False,
    random_state=42,
    eval_metric='logloss'
)

# Hyperparameter tuning for XGBoost
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 1.0],
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model
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

model_file_path = os.path.join(MODEL_DIR, "scalping_model_xgb.pkl")
joblib.dump(best_model, model_file_path)
print(f"Model saved to {model_file_path}")