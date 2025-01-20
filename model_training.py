import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler

# Directory containing the data files
DATA_DIR = "data"
INSTRUMENTS = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "XAU_USD", "XAG_USD"]

# Function to preprocess the data
def preprocess_data(file_path):
    """
    Load and preprocess data for model training.
    """
    data = pd.read_csv(file_path)
    
    # Feature engineering (example features)
    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['Price_Change'] = data['close'].pct_change()  # Percent change in price
    data['RSI'] = compute_rsi(data['close'], window=14)  # Relative Strength Index
    data['MACD'] = compute_macd(data['close'])  # Moving Average Convergence Divergence
    data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, -1)  # 1 for Buy, -1 for Sell
    
    # Add Lag features
    data['Lag_1'] = data['close'].shift(1)  # Previous day's close
    data['Lag_2'] = data['close'].shift(2)  # 2-day lag
    
    # Drop NaN values created by rolling windows
    data.dropna(inplace=True)
    
    # Features and labels
    X = data[['SMA_5', 'SMA_20', 'Price_Change', 'RSI', 'MACD', 'Lag_1', 'Lag_2']]
    y = data['Target']
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Helper functions for indicators
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data, short_window=12, long_window=26):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    return short_ema - long_ema

# Combined dataset for multiple instruments
X_combined = []
y_combined = []

for instrument in INSTRUMENTS:
    file_path = os.path.join(DATA_DIR, f"{instrument}_data.csv")
    if os.path.exists(file_path):
        print(f"Processing data for {instrument}...")
        X, y = preprocess_data(file_path)
        X_combined.append(X)
        y_combined.append(y)
    else:
        print(f"Data file for {instrument} not found. Skipping...")

# Combine all data into single datasets
X_combined = pd.concat(X_combined, axis=0)
y_combined = pd.concat(y_combined, axis=0)

# Check data shapes
print(f"X_combined shape: {X_combined.shape}")
print(f"y_combined shape: {y_combined.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'class_weight': ['balanced', None]
}

print("Tuning hyperparameters with RandomizedSearchCV...")
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                   param_distributions=param_dist,
                                   n_iter=50,  # Increase the number of iterations for better results
                                   cv=5,  # Use 5-fold cross-validation for more reliable results
                                   verbose=2,
                                   n_jobs=-1,  # Use all CPU cores
                                   random_state=42)

# Perform RandomizedSearchCV
random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")

# Train the best model from random search
model = random_search.best_estimator_

# Cross-validation to evaluate model performance
cross_val_scores = cross_val_score(model, X_combined, y_combined, cv=5, n_jobs=-1)
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean score: {cross_val_scores.mean()}")

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
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
joblib.dump(model, model_file_path)
print(f"Model saved to {model_file_path}")