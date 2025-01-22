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
MODEL = joblib.load('trained_model.pkl')

# List of instruments to trade (same as in your model)
INSTRUMENTS = [
    'EUR_USD',  # Euro / US Dollar
    'USD_JPY',  # US Dollar / Japanese Yen
    'GBP_USD',  # British Pound / US Dollar
    'AUD_USD',  # Australian Dollar / US Dollar
    'USD_CHF',  # US Dollar / Swiss Franc
    'EUR_JPY',  # Euro / Japanese Yen
    'GBP_JPY',  # British Pound / Japanese Yen
    'EUR_GBP',  # Euro / British Pound
    'USD_CAD',  # US Dollar / Canadian Dollar
    'NZD_USD'   # New Zealand Dollar / US Dollar
]

def get_account_balance():
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

def create_features(open_prices, high_prices, low_prices, close_prices, volumes, timestamps):
    features = {}
    

    # Price Data
    features['open'] = open_prices[-1] if len(open_prices) >= 1 else np.nan
    features['high'] = high_prices[-1] if len(high_prices) >= 1 else np.nan
    features['low'] = low_prices[-1] if len(low_prices) >= 1 else np.nan
    features['close'] = close_prices[-1] if len(close_prices) >= 1 else np.nan
    features['volume'] = volumes[-1] if len(volumes) >= 1 else np.nan

    # Returns
    features['returns'] = ((close_prices[-1] - close_prices[-2]) / close_prices[-2]) * 100 if len(close_prices) >= 2 else np.nan

    # Volatility
    features['volatility'] = np.std([((close_prices[i] - close_prices[i - 1]) / close_prices[i - 1]) * 100 
                                     for i in range(-10, 0)]) if len(close_prices) >= 10 else np.nan

    # Moving Averages
    features['ma_short'] = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else np.nan
    features['ma_long'] = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.nan
    features['ma_diff'] = features['ma_short'] - features['ma_long'] 

    # Exponential Moving Averages
    ema_short = pd.Series(close_prices).ewm(span=5, adjust=False).mean()
    ema_short_last = ema_short.iloc[-1] if  len(close_prices) >= 5 else np.nan
    print(ema_short)
    ema_long = pd.Series(close_prices).ewm(span=20, adjust=False).mean()[-1] if len(close_prices) >= 20 else np.nan
    features['ema_short'] = ema_short
    features['ema_long'] = ema_long
    features['ema_diff'] = ema_short - ema_long
    

    # Bollinger Bands
    rolling_std = np.std(close_prices[-20:]) if len(close_prices) >= 20 else np.nan
    features['bollinger_upper'] = features['ma_long'] + (2 * rolling_std)
    features['bollinger_lower'] = features['ma_long'] - (2 * rolling_std)
    features['bollinger_bandwidth'] = features['bollinger_upper'] - features['bollinger_lower']
    

    # RSI (Relative Strength Index)
    delta = np.diff(close_prices[-15:]) if len(close_prices) >= 15 else np.array([np.nan])
    gain = np.maximum(delta, 0).mean() if delta.size > 0 else np.nan
    loss = -np.minimum(delta, 0).mean() if delta.size > 0 else np.nan
    rs = gain / loss if loss != 0 else np.nan
    features['rsi'] = 100 - (100 / (1 + rs)) if not np.isnan(rs) else np.nan

    # MACD (Moving Average Convergence Divergence)
    macd = pd.Series(close_prices).ewm(span=12, adjust=False).mean()[-1] - pd.Series(close_prices).ewm(span=26, adjust=False).mean()[-1]
    macd_signal = pd.Series([macd]).ewm(span=9, adjust=False).mean()[-1]
    features['macd'] = macd
    features['macd_signal'] = macd_signal
    features['macd_diff'] = macd - macd_signal

    # Price Differences
    features['high_low_diff'] = high_prices[-1] - low_prices[-1] if len(high_prices) >= 1 and len(low_prices) >= 1 else np.nan
    features['open_close_diff'] = open_prices[-1] - close_prices[-1] if len(open_prices) >= 1 and len(close_prices) >= 1 else np.nan

    # Timestamp and Instrument are placeholders; adjust as needed
    features['timestamp'] = timestamps[-1] if len(timestamps) >= 1 else np.nan
    features['instrument'] = 'placeholder'

    # Future Price is assumed shifted by TARGET_LOOKAHEAD; adjust as needed
    features['future_price'] = close_prices[-1] if len(close_prices) >= 1 else np.nan

    # Convert features into a DataFrame
    features_df = pd.DataFrame([features]).fillna(0)

    # Ensure all 22 features are included
    return features_df[[
        'returns', 'volatility', 'ma_short', 'ma_long', 'ma_diff', 'ema_short', 'ema_long', 'ema_diff',
        'bollinger_upper', 'bollinger_lower', 'bollinger_bandwidth', 'rsi', 'macd', 'macd_signal', 
        'macd_diff', 'high_low_diff', 'open_close_diff', 'instrument', 'timestamp', 'future_price', 'close', 'volume'
    ]]

def calculate_atr(close_prices, high_prices, low_prices, period=14):
    df = pd.DataFrame({'high': high_prices, 'low': low_prices, 'close': close_prices})
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['prev_close']), abs(x['low'] - x['prev_close'])),
        axis=1
    )
    df['atr'] = df['tr'].ewm(span=period, min_periods=1).mean()
    atr_value = df['atr'].iloc[-1]
    return atr_value

def get_confidence(features, prediction):
    try:
        if prediction == 1:
            confidence = MODEL.predict_proba(features)[0][1]
        elif prediction == 0:
            confidence = MODEL.predict_proba(features)[0][0]
        else:
            confidence = 0
        return confidence * 100
    except Exception as e:
        print(f"Error calculating confidence: {e}")
        return 0

def get_instrument_precision(instrument):
    precision_map = {
        "EUR_USD": 5,  # Euro / US Dollar
        "USD_JPY": 3,  # US Dollar / Japanese Yen
        "GBP_USD": 5,  # British Pound / US Dollar
        "AUD_USD": 5,  # Australian Dollar / US Dollar
        "USD_CAD": 5,  # US Dollar / Canadian Dollar
        "USD_CHF": 5,  # US Dollar / Swiss Franc
        "EUR_JPY": 3,  # Euro / Japanese Yen
        "GBP_JPY": 3,  # British Pound / Japanese Yen
        "EUR_GBP": 5,  # Euro / British Pound
        "NZD_USD": 5   # New Zealand Dollar / US Dollar
    }
    return precision_map.get(instrument, 5)  # Default to 5 if precision is unknown

def execute_ioc_order(instrument, side, trade_amount, stop_loss, take_profit, current_price, slippage=0.0005):
    try:
        precision = get_instrument_precision(instrument)
        
        # Round trade_amount to the nearest whole number
        rounded_trade_amount = int(round(trade_amount))
        
        slippage_adjustment = current_price + slippage if side == "buy" else current_price - slippage
        rounded_price = round(slippage_adjustment, precision)
        rounded_stop_loss = round(stop_loss, precision)
        rounded_take_profit = round(take_profit, precision)
        print(f"Order Details - Price: {rounded_price}, SL: {rounded_stop_loss}, TP: {rounded_take_profit}, Units: {rounded_trade_amount}")

        order_payload = {
            "order": {
                "units": rounded_trade_amount if side == "buy" else -rounded_trade_amount,
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
        return f"Market {side} order placed for {instrument}.\n"
    except oandapyV20.exceptions.V20Error as e:
        print(f"Error executing IOC order: {e}")
        return f"Error executing order: {e}"

def execute_trade(instrument):
    try:
        balance = get_account_balance()
        if not balance:
            return "Error: Unable to retrieve account balance."
        trade_amount = balance * 0.01
        market_data = get_latest_data(instrument)
        if not market_data:
            return "Error: Unable to fetch market data."

        print("Creating Features.")
        features = create_features(
            market_data['close_prices'],
            market_data['high_prices'],  # Add high_prices
            market_data['low_prices'],   # Add low_prices
            market_data['close_prices'],
            market_data['volumes'],
            market_data['timestamps']
        )
        prediction = MODEL.predict(features)        [0] 
        prediction_proba = MODEL.predict_proba(features)[0]  # Probabilities for each class
        print(f"Prediction: {prediction}, Buy Probability: {prediction_proba[1]}, Sell Probability: {prediction_proba[0]}")
        atr = calculate_atr(
            market_data['close_prices'],
            market_data['high_prices'],
            market_data['low_prices']
        )
        current_price = market_data['prices']['buy'] if prediction == 1 else market_data['prices']['sell']
        # Set tighter multipliers for SL and        TP for scalping
        stop_loss = current_price - atr * 0.2 if prediction == 1 else current_price + atr * 0.2
        take_profit = current_price + atr * 0.4 if prediction == 1 else current_price - atr * 0.4

        print(f"Instrument: {instrument}, SL: {stop_loss}, TP: {take_profit}, ATR: {atr}")
        confidence = get_confidence(features, prediction)
        if confidence < 70:
            return "Confidence too low to execute trade.\n"

        side = "buy" if prediction == 1 else "sell"
        return execute_ioc_order(instrument, side, trade_amount, stop_loss, take_profit, current_price)
    except Exception as e:
        print(f"Error during trade execution: {e}")
        return "Error during trade execution."

if __name__ == "__main__":
    for instrument in INSTRUMENTS:
        print(execute_trade(instrument))