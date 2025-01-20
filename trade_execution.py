import os
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.orders as orders
import joblib
import numpy as np
from datetime import datetime

# OANDA API credentials (replace with your own)
ACCESS_TOKEN = os.getenv("API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
CLIENT = oandapyV20.API(access_token=ACCESS_TOKEN)

# Load the trained machine learning model (replace 'model.pkl' with your actual model filename)
MODEL = joblib.load('models/trading_model.pkl')

# Instrument to trade
INSTRUMENT = 'EUR_USD'  # You can modify this to any supported pair/commodity

# Risk management parameters
TRADE_AMOUNT = 1000  # Capital to risk per trade (adjust as necessary)
STOP_LOSS = 0.01  # Stop loss in terms of price change (e.g., 0.01 means 1 pip)
TAKE_PROFIT = 0.02  # Take profit in terms of price change (e.g., 0.02 means 2 pips)

# Function to get the latest market data from OANDA (closing price and volume)
def get_latest_data(instrument):
    params = {
        "granularity": "H1",  # 1-hour candles, adjust as needed
        "count": 1  # Fetch the latest data point
    }
    
    # Request data from OANDA
    try:
        response = oandapyV20.endpoints.Instruments.InstrumentsCandles(instrument=instrument, params=params)
        CLIENT.request(response)
        candles = response.response['candles']
        
        # Extract relevant data (close price and volume)
        latest_candle = candles[0]
        close_price = float(latest_candle['mid']['c'])
        volume = int(latest_candle['volume'])
        
        # Return the data as a numpy array for prediction
        return np.array([close_price, volume]).reshape(1, -1)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Function to execute a trade (buy, sell, or hold)
def execute_trade(action, instrument):
    # Fetch the latest data for prediction
    latest_data = get_latest_data(instrument)
    if latest_data is None:
        return
    
    # Predict the action (buy, sell, or hold)
    prediction = MODEL.predict(latest_data)
    print(f"Model predicted action: {prediction[0]}")

    if prediction == 1:  # Buy signal
        print("Placing Buy Order...")
        order = orders.OrderCreate(
            ACCOUNT_ID,
            data={
                "order": {
                    "units": TRADE_AMOUNT,  # Buy order with the specified amount
                    "instrument": instrument,
                    "time_in_force": "GTC",
                    "type": "MARKET",
                    "position_fill": "DEFAULT",
                    "stopLoss": {"price": str(latest_data[0] - STOP_LOSS)},
                    "takeProfit": {"price": str(latest_data[0] + TAKE_PROFIT)},
                }
            }
        )
        CLIENT.request(order)
        print("Buy order placed successfully!")
    
    elif prediction == -1:  # Sell signal
        print("Placing Sell Order...")
        order = orders.OrderCreate(
            ACCOUNT_ID,
            data={
                "order": {
                    "units": -TRADE_AMOUNT,  # Sell order with the specified amount
                    "instrument": instrument,
                    "time_in_force": "GTC",
                    "type": "MARKET",
                    "position_fill": "DEFAULT",
                    "stopLoss": {"price": str(latest_data[0] + STOP_LOSS)},
                    "takeProfit": {"price": str(latest_data[0] - TAKE_PROFIT)},
                }
            }
        )
        CLIENT.request(order)
        print("Sell order placed successfully!")
    
    else:
        print("No action taken. Model predicted 'hold'.")

# Run the trading script
if __name__ == "__main__":
    execute_trade(action=None, instrument=INSTRUMENT)