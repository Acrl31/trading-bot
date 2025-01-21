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

# List of instruments to trade (optimized for scalping)
INSTRUMENTS = [
    'EUR_USD',  # Euro / US Dollar
    'USD_JPY',  # US Dollar / Japanese Yen
    'GBP_USD',  # British Pound / US Dollar
    'AUD_USD',  # Australian Dollar / US Dollar
    'USD_CAD',  # US Dollar / Canadian Dollar
    'USD_CHF',  # US Dollar / Swiss Franc
    'EUR_JPY',  # Euro / Japanese Yen
    'GBP_JPY',  # British Pound / Japanese Yen
    'EUR_GBP',  # Euro / British Pound
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
        params = {"granularity": "M1", "count": 100, "price": "M"}
        request = instruments.InstrumentsCandles(instrument, params=params)
        response = CLIENT.request(request)
        candles = response['candles']
        market_data = {
            'open_prices': [float(c['mid']['o']) for c in candles],
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

    # Moving Averages
    features['ma_short'] = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else np.nan
    features['ma_long'] = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.nan
    
    # Exponential Moving Averages
    features['ema_short'] = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else np.nan
    features['ema_long'] = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.nan
    
    # Bollinger Bands
    rolling_std = np.std(close_prices[-20:]) if len(close_prices) >= 20 else np.nan
    features['bollinger_upper'] = features['ma_long'] + (2 * rolling_std)
    features['bollinger_lower'] = features['ma_long'] - (2 * rolling_std)
    
    # RSI (Relative Strength Index)
    features['rsi'] = np.mean(close_prices[-14:])  # Simplified RSI calculation
    
    # MACD (Moving Average Convergence Divergence)
    features['macd'] = np.mean(close_prices[-12:]) - np.mean(close_prices[-26:])  # Simplified MACD
    features['macd_signal'] = np.mean(close_prices[-9:])  # Simplified Signal Line
    features['macd_diff'] = features['macd'] - features['macd_signal']
    
    # Price Differences
    features['high_low_diff'] = high_prices[-1] - low_prices[-1] if len(high_prices) >= 1 and len(low_prices) >= 1 else np.nan
    features['open_close_diff'] = open_prices[-1] - close_prices[-1] if len(open_prices) >= 1 and len(close_prices) >= 1 else np.nan

    # Convert features into a DataFrame
    features_df = pd.DataFrame([features]).fillna(0)
    
    # Return the relevant columns as requested
    return features_df[['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long', 'ema_short', 
                        'ema_long', 'bollinger_upper', 'bollinger_lower', 'rsi', 'macd', 'macd_signal', 
                        'macd_diff', 'high_low_diff', 'open_close_diff']]

def calculate_atr_scalping(close_prices, high_prices, low_prices, period=5):
    df = pd.DataFrame({'high': high_prices, 'low': low_prices, 'close': close_prices})
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['prev_close']), abs(x['low'] - x['prev_close'])),
        axis=1
    )
    df['atr'] = df['tr'].rolling(window=period).mean()
    atr_value = df['atr'].iloc[-1]
    return atr_value

def get_confidence(features, prediction):
    try:
        prob = MODEL.predict_proba(features)[0]
        if prediction == 1:
            confidence = prob[1]  # Probability of class 1 (positive)
        elif prediction == 0:
            confidence = prob[0]  # Probability of class 0 (negative)
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
        
        # Adjust prices considering slippage
        slippage_adjustment = current_price + slippage if side == "buy" else current_price - slippage
        rounded_price = round(slippage_adjustment, precision)
        
        # Ensure stop loss and take profit are at a reasonable distance
        slippage_stop_loss = stop_loss if side == "buy" else current_price + 0.001
        slippage_take_profit = take_profit if side == "buy" else current_price - 0.001
        
        # Round the prices
        rounded_stop_loss = round(slippage_stop_loss, precision)
        rounded_take_profit = round(slippage_take_profit, precision)
        
        # Ensure there's a minimum difference between stop loss and take profit
        min_distance = 0.0010  # 10 pips for EUR/JPY
        if abs(rounded_stop_loss - rounded_take_profit) < min_distance:
            if side == "buy":
                rounded_take_profit = rounded_stop_loss + min_distance
            else:
                rounded_take_profit = rounded_stop_loss - min_distance
        
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
        print(response)
        return f"Market {side} order placed for {instrument}."
    except oandapyV20.exceptions.V20Error as e:
        print(f"Error executing IOC order: {e}")
        return f"Error executing order: {e}"
        
def execute_trade(instrument):
    data = get_latest_data(instrument)
    
    if data:
        close_prices = data['close_prices']
        high_prices = data['high_prices']
        low_prices = data['low_prices']
        open_prices = data['open_prices']
        volumes = data['volumes']
        current_price = data['prices']['buy']  # Using buy price for now
        
        atr = calculate_atr_scalping(close_prices, high_prices, low_prices)
        print(f"ATR value: {atr}")

        features_df = create_features(open_prices, high_prices, low_prices, close_prices, volumes, data['timestamps'])
        prediction = MODEL.predict(features_df)[0]
        confidence = get_confidence(features_df, prediction)

        if confidence < 60:
            print(f"Low confidence ({confidence}%). Skipping trade for {instrument}.")
            return

        # Calculate stop loss and take profit
        sl_multiplier = 2  # Adjust as necessary
        tp_multiplier = 3  # Adjust as necessary
        stop_loss = current_price - atr * sl_multiplier
        take_profit = current_price + atr * tp_multiplier

        # Ensure stop loss and take profit are far enough from current price
        min_distance = 0.0010  # 10 pips for EUR/JPY
        if abs(stop_loss - current_price) < min_distance:
            stop_loss = current_price + min_distance
        if abs(take_profit - current_price) < min_distance:
            take_profit = current_price + min_distance

        # Execute the order
        execute_ioc_order(instrument, "buy", 1000, stop_loss, take_profit, current_price)  # Adjust trade amount as needed
    else:
        print(f"No data for {instrument}, skipping.")

# Execute trades for each instrument in the list
for instrument in INSTRUMENTS:
    execute_trade(instrument)