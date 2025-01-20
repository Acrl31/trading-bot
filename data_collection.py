import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
from datetime import datetime, timedelta
import os

# OANDA API Credentials
API_KEY = os.getenv("API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"  # Use "https://api-fxtrade.oanda.com/v3" for live trading
oanda_client = oandapyV20.API(access_token=API_KEY)

# List of forex pairs and commodities to collect data for
INSTRUMENTS = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "XAU_USD", "XAG_USD"]

# Timeframe and granularity for historical data
GRANULARITY = "H1"  # 1-hour candles
START_DATE = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
END_DATE = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# Directory to save collected data
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def fetch_historical_data(instrument, start, end, granularity):
    """
    Fetch historical data for a given instrument from OANDA API.
    """
    params = {
        "from": start,
        "to": end,
        "granularity": granularity
    }
    request = instruments.InstrumentsCandles(instrument=instrument, params=params)
    oanda_client.request(request)
    candles = request.response["candles"]
    
    # Parse the response into a DataFrame
    data = [
        {
            "timestamp": candle["time"],
            "open": float(candle["mid"]["o"]),
            "high": float(candle["mid"]["h"]),
            "low": float(candle["mid"]["l"]),
            "close": float(candle["mid"]["c"]),
            "volume": candle["volume"]
        }
        for candle in candles if candle["complete"]
    ]
    return pd.DataFrame(data)

def collect_data():
    """
    Collect historical data for all instruments and save to CSV files in the 'data/' folder.
    """
    for instrument in INSTRUMENTS:
        print(f"Collecting data for {instrument}...")
        data = fetch_historical_data(instrument, START_DATE, END_DATE, GRANULARITY)
        
        # Save data to CSV in the 'data/' folder
        file_path = os.path.join(DATA_DIR, f"{instrument}_data.csv")
        data.to_csv(file_path, index=False)
        print(f"Saved data for {instrument} to {file_path}.")

if __name__ == "__main__":
    collect_data()