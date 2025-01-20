import os
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.orders as orders
import joblib
import numpy as np
import boto3
from datetime import datetime

# OANDA API credentials (replace with your own)
ACCESS_TOKEN = os.getenv("API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
CLIENT = oandapyV20.API(access_token=ACCESS_TOKEN)

# Load the trained machine learning model (replace 'model.pkl' with your actual model filename)
MODEL = joblib.load('models/trading_model.pkl')

# List of instruments to trade (same as in your model)
INSTRUMENTS = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'XAU_USD', 'XAG_USD']

# Risk management parameters
STOP_LOSS = 0.01  # Stop loss in terms of price change (e.g., 0.01 means 1 pip)
TAKE_PROFIT = 0.02  # Take profit in terms of price change (e.g., 0.02 means 2 pips)

# AWS SNS client initialization
sns_client = boto3.client('sns', region_name='eu-west-1')  # Choose the appropriate region for SNS

# Your SNS topic ARN (Amazon Resource Name)
SNS_TOPIC_ARN = 'arn:aws:sns:eu-west-1:183631325876:ForexSNS:02c146ff-f1bf-4a6f-9ab9-eb45510de19c'  # Replace with your SNS Topic ARN

# Function to get account balance from OANDA
def get_account_balance():
    try:
        response = oandapyV20.endpoints.Account.AccountSummary(ACCOUNT_ID)
        CLIENT.request(response)
        balance = float(response.response['account']['balance'])
        return balance
    except Exception as e:
        print(f"Error fetching account balance: {e}")
        return None

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

# Function to send SNS notification
def send_sns_notification(subject, message):
    try:
        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )
        print(f"SNS notification sent! Message ID: {response['MessageId']}")
    except Exception as e:
        print(f"Error sending SNS notification: {e}")

# Function to execute a trade (buy, sell, or hold)
def execute_trade(action, instrument):
    # Fetch the latest data for prediction
    latest_data = get_latest_data(instrument)
    if latest_data is None:
        return
    
    # Get the current account balance
    capital = get_account_balance()
    if capital is None:
        return

    # Calculate trade amount as 1% of current capital
    trade_amount = capital * 0.01  # 1% of available capital
    print(f"Available capital: {capital}, Trade amount: {trade_amount}")
    
    # Predict the action (buy, sell, or hold)
    prediction = MODEL.predict(latest_data)
    print(f"Model predicted action: {prediction[0]}")

    trade_type = ""
    if prediction == 1:  # Buy signal
        print("Placing Buy Order...")
        order = orders.OrderCreate(
            ACCOUNT_ID,
            data={
                "order": {
                    "units": trade_amount,  # Buy order with the calculated amount
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
        trade_type = "BUY"
    
    elif prediction == -1:  # Sell signal
        print("Placing Sell Order...")
        order = orders.OrderCreate(
            ACCOUNT_ID,
            data={
                "order": {
                    "units": -trade_amount,  # Sell order with the calculated amount
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
        trade_type = "SELL"
    
    else:
        print("No action taken. Model predicted 'hold'.")
        trade_type = "HOLD"
    
    # Send SNS notification about the trade
    subject = f"Trade Executed: {trade_type} - {instrument}"
    message = f"A {trade_type} order was executed for {instrument}.\n\nTrade Amount: {trade_amount}\nStop Loss: {STOP_LOSS}\nTake Profit: {TAKE_PROFIT}"
    send_sns_notification(subject, message)

# Run the trading script
if __name__ == "__main__":
    for instrument in INSTRUMENTS:
        execute_trade(action=None, instrument=instrument)