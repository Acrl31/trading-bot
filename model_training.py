import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
from scipy.stats import randint
from concurrent.futures import ProcessPoolExecutor

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

    # Drop NaN values created by rolling windows
    data.dropna(inplace=True)
    
    # Features and labels
    X = data[['SMA_5', 'SMA_20', 'Price_Change']]
    y = data['Target']
    
    return X, y

# Efficiently process data for multiple instruments using parallelism
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

# Reduced parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'class_weight': ['balanced', None]
}

# RandomizedSearchCV with parallelization and fewer folds
print("Tuning hyperparameters with RandomizedSearchCV...")
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                   param_distributions=param_dist,
                                   n_iter=10,  # Limit the number of iterations for faster results
                                   cv=3,  # Use 3-fold cross-validation
                                   verbose=2,
                                   n_jobs=-1,  # Use all CPU cores
                                   random_state=42)

# Perform RandomizedSearchCV
random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")

# Train the best model from random search
model = random_search.best_estimator_

# Cross-validation to evaluate model performance
cross_val_scores = cross_val_score(model, X_combined, y_combined, cv=3, n_jobs=-1)
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