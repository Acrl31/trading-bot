import boto3
import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.account as account
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize AWS SNS client
sns_client = boto3.client('sns', region_name='eu-west-1')
SNS_TOPIC_ARN = 'arn:aws:sns:eu-west-1:183631325876:ForexSNS'

# Initialize OANDA client
API_KEY = os.getenv("API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
client = oandapyV20.API(access_token=API_KEY)

# Load the pre-trained model
MODEL_PATH = 'models/trading_model.pkl'
model = joblib.load(MODEL_PATH)

# Instruments to trade
INSTRUMENTS = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "XAU_USD", "XAG_USD"]

# Fetch account balance
def get_balance():
    account_details = account.AccountDetails(ACCOUNT_ID)
    response = client.request(account_details)
    balance = float(response['account']['balance'])
    return balance

# Calculate ATR manually
def calculate_atr(instrument, period=14):
    params = {
        "granularity": "M1",
        "count": period + 1,  # To get enough data for calculation
    }
    response = client.request(oandapyV20.endpoints.pricing.PricingInfo(instrument=instrument, params=params))
    prices = response['prices']

    high_prices = [float(price['high']) for price in prices]
    low_prices = [float(price['low']) for price in prices]
    close_prices = [float(price['closeBid']) for price in prices]

    # Calculate the True Range (TR) for each period
    tr = [max(high_prices[i] - low_prices[i], abs(high_prices[i] - close_prices[i-1]), abs(low_prices[i] - close_prices[i-1])) 
          for i in range(1, len(prices))]

    # Calculate the ATR as the average of the last 'period' TR values
    atr = np.mean(tr[-period:])
    return atr

# Get the latest market data for the instrument
def get_market_data(instrument):
    params = {
        "instruments": instrument,
    }
    response = client.request(oandapyV20.endpoints.pricing.PricingInfo(instrument=instrument, params=params))
    price = response['prices'][0]
    return {
        'bid': float(price['closeBid']),
        'ask': float(price['closeAsk']),
        'time': price['time'],
    }

# Prepare data for prediction
def prepare_data(instrument):
    # Get historical data for the instrument (this is an example, adjust as needed)
    params = {
        "granularity": "M1",
        "count": 100,  # You can adjust this as needed
    }
    response = client.request(oandapyV20.endpoints.pricing.PricingInfo(instrument=instrument, params=params))
    prices = response['prices']

    # Prepare the necessary features
    data = pd.DataFrame({
        'close': [float(price['closeBid']) for price in prices],
        'high': [float(price['high']) for price in prices],
        'low': [float(price['low']) for price in prices],
        'volume': [float(price['volume']) for price in prices],
    })

    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['Price_Change'] = data['close'].pct_change()
    data['Volatility'] = data['high'] - data['low']
    data['Volume_Change'] = data['volume'].pct_change()

    # Lag features
    data['Lag_Close_1'] = data['close'].shift(1)
    data['Lag_Close_2'] = data['close'].shift(2)
    data['Lag_Volume_1'] = data['volume'].shift(1)

    data.dropna(inplace=True)

    features = data[['SMA_5', 'SMA_20', 'Price_Change', 'Volatility', 'Volume_Change', 'Lag_Close_1', 'Lag_Close_2', 'Lag_Volume_1']]
    return features

# Execute trade based on the model's prediction
def execute_trade(instrument):
    # Get current market data and prepare features
    market_data = get_market_data(instrument)
    features = prepare_data(instrument)
    latest_features = features.iloc[-1].values.reshape(1, -1)

    # Predict using the model
    prediction = model.predict(latest_features)
    confidence = model.predict_proba(latest_features)[0][1]  # Confidence of the 'buy' prediction

    # Skip trade if confidence is below the threshold
    if confidence < 0.7:  # You can adjust this threshold as needed
        return

    # Get account balance and calculate trade size (1% of balance)
    balance = get_balance()
    trade_value = 0.01 * balance  # 1% of the balance

    # Calculate ATR for SL and TP
    atr = calculate_atr(instrument)
    sl = market_data['ask'] - 2 * atr  # Stop loss 2x ATR below ask price
    tp = market_data['ask'] + 2 * atr  # Take profit 2x ATR above ask price

    # Round SL and TP to 5 decimal places
    sl = round(sl, 5)
    tp = round(tp, 5)

    # Place the trade
    order = orders.OrderCreate(
        ACCOUNT_ID,
        data={
            "order": {
                "units": trade_value // market_data['ask'],  # Calculate number of units to trade
                "instrument": instrument,
                "timeInForce": "FOK",
                "type": "LIMIT",
                "price": market_data['ask'],
                "stopLoss": sl,
                "takeProfit": tp,
            }
        }
    )
    client.request(order)

    # Send SNS notification
    sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Message=f"Trade executed for {instrument}.\nType: {'Buy' if prediction == 1 else 'Sell'}\n"
                f"Units: {trade_value // market_data['ask']}\n"
                f"SL: {sl}\nTP: {tp}\nConfidence: {confidence}",
        Subject=f"Trade Notification for {instrument}",
    )

# Main loop to execute trades for each instrument
def run_trading_bot():
    for instrument in INSTRUMENTS:
        execute_trade(instrument)

if __name__ == "__main__":
    run_trading_bot()