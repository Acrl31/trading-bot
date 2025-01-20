import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    
    # Feature engineering (example features)
    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['Price_Change'] = data['close'].pct_change()  # Percent change in price
    data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, -1)  # 1 for Buy, -1 for Sell

    # Handle the last row case where shift would result in NaN
    data.loc[data.index[-1], 'Target'] = data['Target'].iloc[-2]  # Use previous value (1 or -1)

    # Drop NaN values created by rolling windows and percentage change
    data.dropna(inplace=True)
    
    # Features and labels
    X = data[['SMA_5', 'SMA_20', 'Price_Change']]
    y = data['Target']
    
    return X, y

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

# Splitting the data into train and test sets
print("Splitting the data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']  # Adding class weight to handle imbalanced data
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1)
print("Tuning hyperparameters with GridSearchCV...")

# Perform GridSearchCV
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# Train the best model from grid search
model = grid_search.best_estimator_

# Cross-validation to evaluate model performance
cross_val_scores = cross_val_score(model, X_combined, y_combined, cv=5)
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
print(f"Precision: {precision_score(y_test, y_pred, average='binary')}")
print(f"Recall: {recall_score(y_test, y_pred, average='binary')}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='binary')}")

# Save the trained model
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model_file_path = os.path.join(MODEL_DIR, "trading_model.pkl")
joblib.dump(model, model_file_path)
print(f"Model saved to {model_file_path}")