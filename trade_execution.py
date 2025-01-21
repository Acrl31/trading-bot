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

def get_account_balance():
    """
    Fetch the current balance of the account.
    
    Returns:
        float: The account balance.
    """
    try:
        account_request = accounts.AccountDetails(ACCOUNT_ID)
        response = CLIENT.request(account_request)
        balance = float(response['account']['balance'])
        return balance
    except oandapyV20.exceptions.V20Error as e:
        print(f"Error fetching account balance: {e}")
        return None

def get_latest_data(instrument):
    try:
        params = {"granularity": "H1", "count": 100, "price": "M"}
        request = instruments.InstrumentsCandles(instrument, params=params)
        response = CLIENT.request(request)
        candles = response['candles']
        market_data = {
            'close_prices': [float(c['mid']['c']) for c in candles],
            'high_prices': [float(c['mid']['h']) for c in candles],
            'low_prices': [float(c['mid']['l']) for c in candles],
            'volumes': [c['volume'] for c in candles],
            'timestamps': [c['time'] for c in candles],
            'prices': {
                'buy': float(candles[-1]['mid']['c']),
                'sell': float(candles[-1]['mid']['c'])
            }
        }
        return market_data
    except Exception as e:
        print(f"Error fetching data for {instrument}: {e}")
        return None

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

    if timestamps:
        last_timestamp = isoparse(timestamps[-1])
        features['Day_Of_Week'] = last_timestamp.weekday()
        features['Hour_Of_Day'] = last_timestamp.hour
        features['Lag_Hour_1'] = isoparse(timestamps[-2]).hour if len(timestamps) >= 2 else np.nan
    else:
        features['Day_Of_Week'] = features['Hour_Of_Day'] = features['Lag_Hour_1'] = np.nan

    features_df = pd.DataFrame([features]).fillna(0)
    feature_order = [
        'SMA_5', 'SMA_20', 'Price_Change', 'Volatility', 'Volume_Change',
        'Lag_Close_1', 'Lag_Close_2', 'Lag_Volume_1', 'Day_Of_Week',
        'Hour_Of_Day', 'Lag_Hour_1'
    ]
    return features_df[feature_order]

def calculate_atr(close_prices, high_prices, low_prices, period=14):
    df = pd.DataFrame({'high': high_prices, 'low': low_prices, 'close': close_prices})
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['prev_close']), abs(x['low'] - x['prev_close'])),
        axis=1
    )
    df['atr'] = df['tr'].ewm(span=period, min_periods=1).mean()
    return df['atr'].iloc[-1]

def get_confidence(features, prediction):
    try:
        if prediction == 1:
            confidence = MODEL.predict_proba(features)[0][1]
        elif prediction == -1:
            confidence = MODEL.predict_proba(features)[0][0]
        else:
            confidence = 0
        return confidence * 100
    except Exception as e:
        print(f"Error calculating confidence: {e}")
        return 0

def get_instrument_precision(instrument):
    """
    Get the precision allowed for a specific instrument.
    
    Parameters:
        instrument (str): The instrument to trade (e.g., "XAG_USD").
    
    Returns:
        int: The number of decimal places allowed.
    """
    precision_map = {
        "XAU_USD": 2,  # Gold
        "XAG_USD": 3,  # Silver
        "EUR_USD": 5,  # Euro/USD
        "USD_JPY": 3,  # USD/JPY
        "GBP_USD": 5,  # GBP/USD
        "AUD_USD": 5,  # AUD/USD
    }
    return precision_map.get(instrument, 5)  # Default to 5 if not specified

def execute_ioc_order(instrument, side, trade_amount, stop_loss, take_profit, current_price, slippage=0.0002):
    try:
        # Get the instrument's pip size
        precision = get_instrument_precision(instrument)
        pip_size = 10 ** -precision  # For EUR/USD, pip size would be 0.0001 for precision=5

        # Adjust the price for slippage
        if side == "buy":
            slippage_adjustment = current_price + slippage
        else:
            slippage_adjustment = current_price - slippage

        # Round price, stop loss, and take profit to the correct precision
        rounded_price = round(slippage_adjustment, precision)
        rounded_stop_loss = round(stop_loss, precision)
        rounded_take_profit = round(take_profit, precision)

        # Construct the order payload for market order
        order_payload = {
            "order": {
                "units": trade_amount if side == "buy" else -trade_amount,
                "instrument": instrument,
                "timeInForce": "IOC",  # Immediate Or Cancel
                "type": "MARKET",
                "stopLossOnFill": {"price": str(rounded_stop_loss)},
                "takeProfitOnFill": {"price": str(rounded_take_profit)},
                "positionFill": "DEFAULT",
            }
        }

        # Send the request for market order
        r = orders.OrderCreate(ACCOUNT_ID, data=order_payload)
        response = CLIENT.request(r)

        return f"Market {side} order placed for {instrument} at price {rounded_price} with SL {rounded_stop_loss} and TP {rounded_take_profit}."

    except oandapyV20.exceptions.V20Error as e:
        print(f"Error executing IOC order: {e}")
        return f"Error executing order: {e}"

def execute_trade(instrument):
    try:
        balance = get_account_balance()
        if not balance:
            return "Error: Unable to retrieve account balance."
        
        # Ensure that the trade size is at most 1% of the balance
        trade_amount = balance * 0.01
        
        market_data = get_latest_data(instrument)
        if not market_data:
            return "Error: Unable to fetch market data."

        features = create_features(
            market_data['close_prices'],
            market_data['volumes'],
            market_data['timestamps']
        )
        prediction = MODEL.predict(features)[0]
        atr = calculate_atr(
            market_data['close_prices'],
            market_data['high_prices'],
            market_data['low_prices']
        )
        stop_loss = round(atr * 2, 5)
        take_profit = round(atr * 4, 5)
        current_price = market_data['prices']['buy'] if prediction == 1 else market_data['prices']['sell']
        confidence = get_confidence(features, prediction)
        
        if confidence < 30:
            return "Confidence too low to execute trade."

        side = "buy" if prediction == 1 else "sell"
        return execute_ioc_order(instrument, side, trade_amount, stop_loss, take_profit, current_price)
    except Exception as e:
        print(f"Error during trade execution: {e}")
        return "Error during trade execution."

if __name__ == "__main__":
    for instrument in INSTRUMENTS:
        print(execute_trade(instrument))