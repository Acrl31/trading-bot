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
import boto3  # AWS SDK for SNS

# OANDA API credentials (replace with your own)
ACCESS_TOKEN = os.getenv("API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
CLIENT = oandapyV20.API(access_token=ACCESS_TOKEN)

# Load the trained machine learning model (replace 'model.pkl' with your actual model filename)
MODEL = joblib.load('models/trading_model.pkl')

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

# SNS client for notifications
sns_client = boto3.client('sns', region_name='eu-west-1')  # Adjust region as needed
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")  # Ensure SNS_TOPIC_ARN is set in your environment

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
    atr_value = df['atr'].iloc[-1]
    return atr_value

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
    precision_map = {
        "EUR_USD": 5,  # Euro / US Dollar
        "USD_JPY": 3,  # US Dollar / Japanese Yen
        "GBP_USD": 5,  # British Pound / US Dollar
        "AUD_USD": 5,  # Australian Dollar / US Dollar
        "USD_CHF": 5,  # US Dollar / Swiss Franc
        "EUR_JPY": 3,  # Euro / Japanese Yen
        "GBP_JPY": 3,  # British Pound / Japanese Yen
        "EUR_GBP": 5,  # Euro / British Pound
        "USD_CAD": 5,  # US Dollar / Canadian Dollar
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
        print(f"Order Response: {response}")
        
        # Create trade message
        order_message = f"{side.capitalize()} order placed for {instrument}. {rounded_trade_amount} units @ {rounded_price}. SL: {rounded_stop_loss}, TP: {rounded_take_profit}"
        return order_message
    except oandapyV20.exceptions.V20Error as e:
        print(f"Error executing IOC order: {e}")
        return f"Error executing order: {e}"

def execute_trade(instrument):
    trade_details = []
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

        current_price = market_data['prices']['buy'] if prediction == 1 else market_data['prices']['sell']
        # Set tighter multipliers for SL and TP for scalping
        stop_loss = current_price - atr * 0.1 if prediction == 1 else current_price + atr * 0.1
        take_profit = current_price + atr * 0.2 if prediction == 1 else current_price - atr * 0.2

        print(f"Instrument: {instrument}, SL: {stop_loss}, TP: {take_profit}, ATR: {atr}")
        confidence = get_confidence(features, prediction)
        if confidence < 70:
            return "Confidence too low to execute trade."

        side = "buy" if prediction == 1 else "sell"
        order_status = execute_ioc_order(instrument, side, trade_amount, stop_loss, take_profit, current_price)
        trade_details.append(order_status)
    except Exception as e:
        print(f"Error during trade execution: {e}")
        return "Error during trade execution."

    return trade_details

def send_sns_notification(trade_details):
    message = "\n".join(trade_details)
    sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Message=message,
        Subject="Trading Orders Summary"
    )
    print("SNS notification sent.")

if __name__ == "__main__":
    all_trade_details = []
    for instrument in INSTRUMENTS:
        trade_result = execute_trade(instrument)
        if isinstance(trade_result, list):
            all_trade_details.extend(trade_result)

    if all_trade_details:
        send_sns_notification(all_trade_details)