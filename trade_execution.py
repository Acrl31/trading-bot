import os
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.endpoints.instruments import InstrumentsCandles
import joblib
import numpy as np

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
        # Get account information from OANDA
        response = CLIENT.request(accounts.AccountSummary(ACCOUNT_ID))
        balance = float(response['account']['balance'])  # Get the balance from the account
        return balance
    except Exception as e:
        print(f"Error fetching account balance: {e}")
        return None

# Function to get the latest market data from OANDA (close, high, and low prices)
def get_latest_data(instrument):
    params = {
        "granularity": "H1",  # 1-hour candles, adjust as needed
        "count": 100  # Fetch the latest 100 data points for volatility calculation
    }
    
    # Request data from OANDA
    try:
        response = InstrumentsCandles(instrument=instrument, params=params)
        CLIENT.request(response)
        candles = response.response['candles']
        
        # Extract relevant data (close, high, and low prices)
        close_prices = [float(candle['mid']['c']) for candle in candles]
        high_prices = [float(candle['mid']['h']) for candle in candles]
        low_prices = [float(candle['mid']['l']) for candle in candles]
        
        return np.array(close_prices), np.array(high_prices), np.array(low_prices)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None

# Calculate the Average True Range (ATR) for an instrument (to use for stop loss and take profit)
def calculate_atr(high_prices, low_prices, close_prices, window=14):
    if high_prices is None or low_prices is None or close_prices is None or len(close_prices) < window:
        return None
    
    # Calculate True Range (TR) values
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - close_prices[:-1]),
                               np.abs(low_prices[1:] - close_prices[:-1])))
    atr = np.mean(tr[-window:])
    return atr

# Function to execute a trade (buy, sell, or hold)
def execute_trade(instrument):
    # Fetch the available balance
    balance = get_balance()
    if balance is None:
        return
    
    # Calculate trade amount (1% of available balance)
    trade_amount = balance * 0.01  # 1% of available balance

    # Fetch market data for ATR calculation and model prediction
    close_prices, high_prices, low_prices = get_latest_data(instrument)
    if close_prices is None or high_prices is None or low_prices is None:
        return

    # Calculate ATR for dynamic stop loss and take profit
    atr = calculate_atr(high_prices, low_prices, close_prices)
    if atr is None:
        return

    # Dynamically adjust stop loss and take profit based on ATR
    stop_loss = atr * 2  # Example: stop loss is 2x ATR
    take_profit = atr * 4  # Example: take profit is 4x ATR

    # Fetch the latest data for prediction
    latest_data = close_prices[-1].reshape(1, -1)  # Only use the last data point
    if latest_data is None:
        return
    
    # Predict the action (buy, sell, or hold)
    prediction = MODEL.predict(latest_data)
    print(f"Model predicted action for {instrument}: {prediction[0]}")

    # Only trade if the model's confidence is high enough
    if prediction == 1:  # Buy signal
        print(f"Placing Buy Order for {instrument}...")
        order = orders.OrderCreate(
            ACCOUNT_ID,
            data={
                "order": {
                    "units": trade_amount,  # Buy order with the specified amount
                    "instrument": instrument,
                    "time_in_force": "GTC",
                    "type": "MARKET",
                    "position_fill": "DEFAULT",
                    "stopLoss": {"price": str(close_prices[-1] - stop_loss)},
                    "takeProfit": {"price": str(close_prices[-1] + take_profit)},
                }
            }
        )
        CLIENT.request(order)
        print(f"Buy order placed for {instrument} successfully!")
    
    elif prediction == -1:  # Sell signal
        print(f"Placing Sell Order for {instrument}...")
        order = orders.OrderCreate(
            ACCOUNT_ID,
            data={
                "order": {
                    "units": -trade_amount,  # Sell order with the specified amount
                    "instrument": instrument,
                    "time_in_force": "GTC",
                    "type": "MARKET",
                    "position_fill": "DEFAULT",
                    "stopLoss": {"price": str(close_prices[-1] + stop_loss)},
                    "takeProfit": {"price": str(close_prices[-1] - take_profit)},
                }
            }
        )
        CLIENT.request(order)
        print(f"Sell order placed for {instrument} successfully!")
    
    else:
        print(f"No action taken for {instrument}. Model predicted 'hold'.")

# Run the trading script for all instruments
if __name__ == "__main__":
    for instrument in INSTRUMENTS:
        execute_trade(instrument)