import os
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import joblib
import numpy as np
from datetime import datetime
from dateutil.parser import isoparse

# OANDA API credentials (replace with your own)
ACCESS_TOKEN = os.getenv("API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
CLIENT = oandapyV20.API(access_token=ACCESS_TOKEN)

# Load the trained machine learning model (replace 'model.pkl' with your actual model filename)
MODEL = joblib.load('models/trading_model.pkl')

# List of instruments to trade (same as in your model)
INSTRUMENTS = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'XAU_USD', 'XAG_USD']

# Function to get the available balance from OANDA
def get_balance():
    try:
        response = CLIENT.request(accounts.AccountSummary(ACCOUNT_ID))
        balance = float(response['account']['balance'])
        return balance
    except Exception as e:
        print(f"Error fetching account balance: {e}")
        return None

# Function to get the latest market data from OANDA
def get_latest_data(instrument):
    params = {
        "granularity": "H1",  # 1-hour candles
        "count": 100  # Fetch the latest 100 data points
    }
    try:
        req = instruments.InstrumentsCandles(instrument=instrument, params=params)
        CLIENT.request(req)
        candles = req.response['candles']
        close_prices = [float(c['mid']['c']) for c in candles if c['complete']]
        volumes = [float(c['volume']) for c in candles if c['complete']]
        timestamps = [isoparse(c['time']) for c in candles if c['complete']]  # Use isoparse for flexible parsing
        return close_prices, volumes, timestamps
    except Exception as e:
        print(f"Error fetching data for {instrument}: {e}")
        return None, None, None

# Function to compute features for model prediction
def create_features(close_prices, volumes, timestamps):
    features = {}
    features['SMA_5'] = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else np.nan
    features['SMA_20'] = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.nan
    features['Price_Change'] = ((close_prices[-1] - close_prices[-2]) / close_prices[-2]) * 100 if len(close_prices) >= 2 else np.nan
    features['Volatility'] = np.std(close_prices[-20:]) if len(close_prices) >= 20 else np.nan
    features['Volume_Change'] = ((volumes[-1] - volumes[-2]) / volumes[-2]) * 100 if len(volumes) >= 2 else np.nan
    features['Lag_Close_1'] = close_prices[-2] if len(close_prices) >= 2 else np.nan
    features['Lag_Close_2'] = close_prices[-3] if len(close_prices) >= 3 else np.nan
    features['Lag_Volume_1'] = volumes[-2] if len(volumes) >= 2 else np.nan
    last_timestamp = timestamps[-1] if timestamps else datetime.now()
    features['Day_Of_Week'] = last_timestamp.weekday()
    features['Hour_Of_Day'] = last_timestamp.hour
    features['Lag_Hour_1'] = timestamps[-2].hour if len(timestamps) >= 2 else np.nan
    return pd.DataFrame([features])

# Function to execute a trade
def execute_trade(instrument):
    balance = get_balance()
    if balance is None:
        return
    
    trade_amount = balance * 0.01  # 1% of balance
    close_prices, volumes, timestamps = get_latest_data(instrument)
    if not close_prices or not volumes or not timestamps:
        return

    features = create_features(close_prices, volumes, timestamps)
    if features.isnull().any().any():
        print(f"Insufficient data for {instrument}")
        return

    prediction = MODEL.predict(features)
    print(f"Model predicted action for {instrument}: {prediction[0]}")

    atr = np.std(close_prices[-20:]) if len(close_prices) >= 20 else None
    if atr is None:
        print(f"ATR not available for {instrument}")
        return

    stop_loss = atr * 2
    take_profit = atr * 4
    last_price = close_prices[-1]

    if prediction == 1:  # Buy
        print(f"Placing Buy Order for {instrument}...")
        order = orders.OrderCreate(
            ACCOUNT_ID,
            data={
                "order": {
                    "units": trade_amount,
                    "instrument": instrument,
                    "time_in_force": "GTC",
                    "type": "MARKET",
                    "position_fill": "DEFAULT",
                    "stopLossOnFill": {"price": str(last_price - stop_loss)},
                    "takeProfitOnFill": {"price": str(last_price + take_profit)},
                }
            }
        )
        CLIENT.request(order)
        print(f"Buy order placed for {instrument}.")
    elif prediction == -1:  # Sell
        print(f"Placing Sell Order for {instrument}...")
        order = orders.OrderCreate(
            ACCOUNT_ID,
            data={
                "order": {
                    "units": -trade_amount,
                    "instrument": instrument,
                    "time_in_force": "GTC",
                    "type": "MARKET",
                    "position_fill": "DEFAULT",
                    "stopLossOnFill": {"price": str(last_price + stop_loss)},
                    "takeProfitOnFill": {"price": str(last_price - take_profit)},
                }
            }
        )
        CLIENT.request(order)
        print(f"Sell order placed for {instrument}.")
    else:
        print(f"No action taken for {instrument}.")

# Main loop
if __name__ == "__main__":
    for instrument in INSTRUMENTS:
        execute_trade(instrument)