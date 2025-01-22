import os
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import joblib
import numpy as np
from datetime import datetime

# OANDA API credentials (replace with your own)
ACCESS_TOKEN = os.getenv("API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
CLIENT = oandapyV20.API(access_token=ACCESS_TOKEN)

# Load the trained machine learning model
MODEL = joblib.load('trained_model.pkl')

# Instruments to trade
INSTRUMENTS = [
    'EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CHF',
    'EUR_JPY', 'GBP_JPY', 'EUR_GBP', 'USD_CAD', 'NZD_USD'
]

def get_account_balance():
    try:
        account_request = accounts.AccountDetails(ACCOUNT_ID)
        response = CLIENT.request(account_request)
        return float(response['account']['balance'])
    except oandapyV20.exceptions.V20Error as e:
        print(f"Error fetching account balance: {e}")
        return None

def get_latest_data(instrument):
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
            'timestamps': [c['time'] for c in candles],
            'prices': {
                'buy': float(candles[-1]['mid']['c']),
                'sell': float(candles[-1]['mid']['c'])
            }
        }
    except Exception as e:
        print(f"Error fetching data for {instrument}: {e}")
        return None

def create_features(data):
    close_prices = data['close_prices']
    high_prices = data['high_prices']
    low_prices = data['low_prices']
    volumes = data['volumes']
    timestamps = data['timestamps']

    features = {
        'ma_short': np.mean(close_prices[-5:]) if len(close_prices) >= 5 else np.nan,
        'ma_long': np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.nan,
        'ema_short': pd.Series(close_prices).ewm(span=5, adjust=False).mean().iloc[-1] if len(close_prices) >= 5 else np.nan,
        'ema_long': pd.Series(close_prices).ewm(span=20, adjust=False).mean().iloc[-1] if len(close_prices) >= 20 else np.nan,
        'bollinger_upper': (np.mean(close_prices[-20:]) + 2 * np.std(close_prices[-20:])) if len(close_prices) >= 20 else np.nan,
        'bollinger_lower': (np.mean(close_prices[-20:]) - 2 * np.std(close_prices[-20:])) if len(close_prices) >= 20 else np.nan,
        'rsi': calculate_rsi(close_prices),
        'macd': calculate_macd(close_prices)[0],
        'macd_signal': calculate_macd(close_prices)[1],
        'returns': ((close_prices[-1] - close_prices[-2]) / close_prices[-2]) * 100 if len(close_prices) >= 2 else np.nan
    }
    return pd.DataFrame([features]).fillna(0)

def calculate_rsi(prices, period=14):
    delta = np.diff(prices[-(period + 1):]) if len(prices) >= period else np.array([np.nan])
    gain = np.maximum(delta, 0).mean() if delta.size > 0 else np.nan
    loss = -np.minimum(delta, 0).mean() if delta.size > 0 else np.nan
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs)) if not np.isnan(rs) else np.nan

def calculate_macd(prices):
    short_ema = pd.Series(prices).ewm(span=12, adjust=False).mean()
    long_ema = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1]

def execute_trade(instrument):
    try:
        balance = get_account_balance()
        if not balance:
            return "Error: Unable to retrieve account balance."
        
        data = get_latest_data(instrument)
        if not data:
            return f"Error fetching market data for {instrument}."

        features = create_features(data)
        prediction = MODEL.predict(features)[0]
        confidence = MODEL.predict_proba(features)[0][prediction] * 100

        if confidence < 70:
            print(f"Confidence too low for {instrument}: {confidence}%")
            return

        trade_amount = balance * 0.01
        current_price = data['prices']['buy'] if prediction == 1 else data['prices']['sell']
        stop_loss, take_profit = calculate_trade_parameters(current_price, data, prediction)

        return place_order(instrument, "buy" if prediction == 1 else "sell", trade_amount, stop_loss, take_profit)
    except Exception as e:
        print(f"Error executing trade for {instrument}: {e}")
        return None

def calculate_trade_parameters(current_price, data, prediction):
    atr = calculate_atr(data['close_prices'], data['high_prices'], data['low_prices'])
    if prediction == 1:
        return current_price - atr * 0.2, current_price + atr * 0.4
    else:
        return current_price + atr * 0.2, current_price - atr * 0.4

def calculate_atr(close_prices, high_prices, low_prices, period=14):
    df = pd.DataFrame({'high': high_prices, 'low': low_prices, 'close': close_prices})
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['prev_close']), abs(x['low'] - x['prev_close'])),
        axis=1
    )
    return df['tr'].rolling(period).mean().iloc[-1]

def place_order(instrument, side, trade_amount, stop_loss, take_profit):
    try:
        units = int(round(trade_amount))
        order = {
            "order": {
                "units": units if side == "buy" else -units,
                "instrument": instrument,
                "timeInForce": "FOK",
                "type": "MARKET",
                "stopLossOnFill": {"price": str(stop_loss)},
                "takeProfitOnFill": {"price": str(take_profit)},
                "positionFill": "DEFAULT"
            }
        }
        r = orders.OrderCreate(ACCOUNT_ID, data=order)
        response = CLIENT.request(r)
        print(f"Order placed for {instrument}: {response}")
        return response
    except oandapyV20.exceptions.V20Error as e:
        print(f"API Error for {instrument}: {e}")
    except Exception as e:
        print(f"Error placing order for {instrument}: {e}")

if __name__ == "__main__":
    for instrument in INSTRUMENTS:
        execute_trade(instrument)