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

# Load the trained machine learning model
MODEL = joblib.load('models/trading_model.pkl')

# List of instruments to trade (gold and silver removed)
INSTRUMENTS = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD']

def get_account_balance():
    """
    Fetch the current balance of the account.
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
    """
    Fetch the latest market data for the given instrument.
    """
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
    """
    Create feature set for the model, ensuring all required features are included.
    """
    features = {
        'SMA_5': np.mean(close_prices[-5:]) if len(close_prices) >= 5 else np.nan,
        'SMA_20': np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.nan,
        'Price_Change': ((close_prices[-1] - close_prices[-2]) / close_prices[-2]) * 100 if len(close_prices) >= 2 else np.nan,
        'Volatility': np.std(close_prices[-20:]) if len(close_prices) >= 20 else np.nan,
        'Volume_Change': ((volumes[-1] - volumes[-2]) / volumes[-2]) * 100 if len(volumes) >= 2 else np.nan,
        'Lag_Close_1': close_prices[-2] if len(close_prices) >= 2 else np.nan,
        'Lag_Close_2': close_prices[-3] if len(close_prices) >= 3 else np.nan,
        'Lag_Volume_1': volumes[-2] if len(volumes) >= 2 else np.nan,
        'Lag_Hour_1': isoparse(timestamps[-2]).hour if len(timestamps) >= 2 else np.nan,  # New feature
    }
    if timestamps:
        last_timestamp = isoparse(timestamps[-1])
        features['Day_Of_Week'] = last_timestamp.weekday()
        features['Hour_Of_Day'] = last_timestamp.hour
    else:
        features['Day_Of_Week'] = features['Hour_Of_Day'] = np.nan

    return pd.DataFrame([features]).fillna(0)

def calculate_atr(close_prices, high_prices, low_prices, period=14):
    """
    Calculate the Average True Range (ATR) for the given data.
    """
    df = pd.DataFrame({'high': high_prices, 'low': low_prices, 'close': close_prices})
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['prev_close']), abs(x['low'] - x['prev_close'])),
        axis=1
    )
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df['atr'].iloc[-1]

def get_instrument_precision(instrument):
    """
    Return the precision for a specific instrument.
    """
    precision_map = {
        "EUR_USD": 5, "USD_JPY": 3, "GBP_USD": 5, "AUD_USD": 5
    }
    return precision_map.get(instrument, 5)

def execute_ioc_order(instrument, side, trade_amount, stop_loss, take_profit, current_price):
    """
    Execute an IOC order with the specified parameters.
    """
    try:
        precision = get_instrument_precision(instrument)
        rounded_stop_loss = round(stop_loss, precision)
        rounded_take_profit = round(take_profit, precision)

        order_payload = {
            "order": {
                "units": trade_amount if side == "buy" else -trade_amount,
                "instrument": instrument,
                "timeInForce": "IOC",
                "type": "MARKET",
                "stopLossOnFill": {"price": str(rounded_stop_loss)},
                "takeProfitOnFill": {"price": str(rounded_take_profit)},
                "positionFill": "DEFAULT",
            }
        }

        r = orders.OrderCreate(ACCOUNT_ID, data=order_payload)
        response = CLIENT.request(r)
        return f"Order placed: {response}"
    except oandapyV20.exceptions.V20Error as e:
        return f"Error executing order: {e}"

def execute_trade(instrument):
    """
    Execute a trade based on model predictions and ATR.
    """
    try:
        balance = get_account_balance()
        if not balance:
            return "Error: Unable to retrieve account balance."

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
        stop_loss = atr * 1.5  # Smaller ATR multiplier
        take_profit = atr * 2.5  # Smaller ATR multiplier
        current_price = market_data['prices']['buy'] if prediction == 1 else market_data['prices']['sell']

        side = "buy" if prediction == 1 else "sell"
        return execute_ioc_order(instrument, side, trade_amount, stop_loss, take_profit, current_price)
    except Exception as e:
        return f"Error during trade execution: {e

Thanks for clarifying your feature set. Based on this, I'll ensure the script consistently generates all these features for both training and live prediction.

Here's the **updated script** to align the live feature generation with your feature set:

---

### Full Script with Correct Features

```python
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

# OANDA API credentials
ACCESS_TOKEN = os.getenv("API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
CLIENT = oandapyV20.API(access_token=ACCESS_TOKEN)

# Load the trained machine learning model
MODEL = joblib.load('models/trading_model.pkl')

# Instruments to trade
INSTRUMENTS = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD']

def get_account_balance():
    """Fetch the current account balance."""
    try:
        account_request = accounts.AccountDetails(ACCOUNT_ID)
        response = CLIENT.request(account_request)
        return float(response['account']['balance'])
    except oandapyV20.exceptions.V20Error as e:
        print(f"Error fetching account balance: {e}")
        return None

def get_latest_data(instrument):
    """Fetch the latest market data for the specified instrument."""
    try:
        params = {"granularity": "H1", "count": 100, "price": "M"}
        request = instruments.InstrumentsCandles(instrument, params=params)
        response = CLIENT.request(request)
        candles = response['candles']
        return {
            'close_prices': [float(c['mid']['c']) for c in candles],
            'high_prices': [float(c['mid']['h']) for c in candles],
            'low_prices': [float(c['mid']['l']) for c in candles],
            'volumes': [c['volume'] for c in candles],
            'timestamps': [c['time'] for c in candles]
        }
    except Exception as e:
        print(f"Error fetching data for {instrument}: {e}")
        return None

def create_features(close_prices, high_prices, low_prices, volumes, timestamps):
    """Generate all required features for the model."""
    features = {
        'SMA_5': np.mean(close_prices[-5:]) if len(close_prices) >= 5 else np.nan,
        'SMA_20': np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.nan,
        'Price_Change': ((close_prices[-1] - close_prices[-2]) / close_prices[-2]) * 100 if len(close_prices) >= 2 else np.nan,
        'Volatility': np.std(close_prices[-20:]) if len(close_prices) >= 20 else np.nan,
        'Volume_Change': ((volumes[-1] - volumes[-2]) / volumes[-2]) * 100 if len(volumes) >= 2 else np.nan,
        'Lag_Close_1': close_prices[-2] if len(close_prices) >= 2 else np.nan,
        'Lag_Close_2': close_prices[-3] if len(close_prices) >= 3 else np.nan,
        'Lag_Volume_1': volumes[-2] if len(volumes) >= 2 else np.nan,
        'Day_Of_Week': isoparse(timestamps[-1]).weekday() if timestamps else np.nan,
        'Hour_Of_Day': isoparse(timestamps[-1]).hour if timestamps else np.nan,
        'Lag_Hour_1': isoparse(timestamps[-2]).hour if len(timestamps) >= 2 else np.nan
    }
    return pd.DataFrame([features]).fillna(0)

def calculate_atr(close_prices, high_prices, low_prices, period=14):
    """Calculate Average True Range (ATR)."""
    df = pd.DataFrame({'high': high_prices, 'low': low_prices, 'close': close_prices})
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['prev_close']), abs(x['low'] - x['prev_close'])),
        axis=1
    )
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df['atr'].iloc[-1]

def execute_order(instrument, side, trade_amount, stop_loss, take_profit):
    """Execute a trade order."""
    try:
        order_payload = {
            "order": {
                "units": trade_amount if side == "buy" else -trade_amount,
                "instrument": instrument,
                "timeInForce": "IOC",
                "type": "MARKET",
                "stopLossOnFill": {"price": str(round(stop_loss, 5))},
                "takeProfitOnFill": {"price": str(round(take_profit, 5))},
                "positionFill": "DEFAULT"
            }
        }
        r = orders.OrderCreate(ACCOUNT_ID, data=order_payload)
        response = CLIENT.request(r)
        return f"Order placed: {response}"
    except oandapyV20.exceptions.V20Error as e:
        return f"Error executing order: {e}"

def execute_trade(instrument):
    """Perform a trade based on model prediction and ATR."""
    try:
        balance = get_account_balance()
        if not balance:
            return "Error: Unable to fetch account balance."

        trade_amount = int(balance * 0.01)
        market_data = get_latest_data(instrument)
        if not market_data:
            return f"Error: Unable to fetch market data for {instrument}."

        features = create_features(
            market_data['close_prices'],
            market_data['high_prices'],
            market_data['low_prices'],
            market_data['volumes'],
            market_data['timestamps']
        )

        prediction = MODEL.predict(features)[0]
        atr = calculate_atr(
            market_data['close_prices'],
            market_data['high_prices'],
            market_data['low_prices']
        )

        stop_loss = atr * 1.5
        take_profit = atr * 2.5
        side = "buy" if prediction == 1 else "sell"
        return execute_order(instrument, side, trade_amount, stop_loss, take_profit)
    except Exception as e:
        return f"Error during trade execution: {e}"

if __name__ == "__main__":
    for instrument in INSTRUMENTS:
        result = execute_trade(instrument)
        print(result)