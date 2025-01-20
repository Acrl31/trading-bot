import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Directory containing the data files
DATA_DIR = "data"
INSTRUMENTS = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "XAU_USD", "XAG_USD"]

# Function to preprocess the data
def preprocess_data(file_path):
    """
    Load and preprocess data for model training, including new features like EMA and RSI.
    """
    data = pd.read_csv(file_path)
    
    # Feature engineering
    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['Price_Change'] = data['close'].pct_change()  # Percent change in price
    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()  # Exponential Moving Average
    data['RSI'] = compute_rsi(data['close'], 14)  # Relative Strength Index (14-period)
    data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, -1)  # 1 for Buy, -1 for Sell

    # Drop NaN values created by rolling windows and other calculations
    data.dropna(inplace=True)
    
    # Features and labels
    X = data[['SMA_5', 'SMA_20', 'Price_Change', 'EMA_12', 'RSI']]
    y = data['Target']
    
    return X, y

# Function to calculate RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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

# Check for class imbalance and apply SMOTE if necessary
print("Checking for class imbalance...")
class_counts = y_combined.value_counts()
print(f"Class distribution:\n{class_counts}")

# Optional: Apply SMOTE if imbalance is detected
smote = SMOTE(random_state=42)
X_combined_resampled, y_combined_resampled = smote.fit_resample(X_combined, y_combined)

# Train-test split
print("Splitting the data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_combined_resampled, y_combined_resampled, test_size=0.2, random_state=42)

# Simplified parameter grid for testing
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]  # Adding class weight to handle imbalanced data
}

# Use StratifiedKFold for better cross-validation with imbalanced data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV with parallel execution
print("Tuning hyperparameters with GridSearchCV...")
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=skf, 
                           n_jobs=-1, 
                           verbose=2)
grid_search.fit(X_train, y_train)

# Best model from grid search
print(f"Best parameters: {grid_search.best_params_}")
model = grid_search.best_estimator_

# Cross-validation to evaluate model performance
print("Evaluating model performance with cross-validation...")
cross_val_scores = cross_val_score(model, X_combined_resampled, y_combined_resampled, cv=skf)
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean score: {cross_val_scores.mean()}")

# Train the best model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model performance on test data...")
y_pred = model.predict(X_test)
print("Model Performance:")
print(classification_report(y_test, y_pred, target_names=["Sell", "Buy"]))
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