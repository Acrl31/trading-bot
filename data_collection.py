import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

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

def fetch_historical_data(instrument, start_date, end_date, granularity):
    """
    Fetch historical data for a given instrument from OANDA API and include relevant features.
    """
    data = []
    current_start = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")

    while current_start < datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ"):
        next_start = current_start + timedelta(days=5)  # Fetch 5 days of data at a time
        params = {
            "from": current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": next_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": granularity,
        }

        try:
            logging.info(f"Fetching data for {instrument}: {params['from']} to {params['to']}")
            request = instruments.InstrumentsCandles(instrument=instrument, params=params)
            oanda_client.request(request)
            candles = request.response.get("candles", [])

            for candle in candles:
                if candle["complete"]:
                    data.append({
                        "time": candle["time"],
                        "bid_open": float(candle["bid"]["o"]),
                        "bid_high": float(candle["bid"]["h"]),
                        "bid_low": float(candle["bid"]["l"]),
                        "bid_close": float(candle["bid"]["c"]),
                        "ask_open": float(candle["ask"]["o"]),
                        "ask_high": float(candle["ask"]["h"]),
                        "ask_low": float(candle["ask"]["l"]),
                        "ask_close": float(candle["ask"]["c"]),
                        "mid_close": float(candle["mid"]["c"]),
                        "volume": candle["volume"],
                    })
        except Exception as e:
            logging.error(f"Error fetching data for {instrument}: {e}")
            break

        current_start = next_start

    # Convert to DataFrame
    return pd.DataFrame(data)

def preprocess_data(data):
    """
    Add features for modeling and filter only the most relevant ones.
    """
    # Calculate spreads
    data["spread"] = data["ask_close"] - data["bid_close"]

    # Calculate volatility (as percentage of mid_close)
    data["volatility"] = (data["ask_high"] - data["bid_low"]) / data["mid_close"]

    # Momentum: difference between current and previous mid_close
    data["momentum"] = data["mid_close"].diff()

    # Volume-weighted average price (VWAP)
    data["VWAP"] = (
        (data["volume"] * (data["ask_close"] + data["bid_close"]) / 2).cumsum()
        / data["volume"].cumsum()
    )

    # Simple Moving Averages
    data["SMA_5"] = data["mid_close"].rolling(window=5).mean()
    data["SMA_20"] = data["mid_close"].rolling(window=20).mean()

    # Exponential Moving Average
    data["EMA_10"] = data["mid_close"].ewm(span=10).mean()

    # Target variable: Buy (1) if the next close is higher, Sell (-1) otherwise
    data["Target"] = (data["mid_close"].shift(-1) > data["mid_close"]).astype(int)

    # Drop rows with missing values (due to rolling calculations)
    data.dropna(inplace=True)

    # Select relevant columns
    return data[[
        "spread", "volatility", "momentum", "VWAP", "SMA_5", "SMA_20", "EMA_10", "Target"
    ]]

def collect_and_save_data():
    """
    Collect, preprocess, and save data for all instruments.
    """
    for instrument in INSTRUMENTS:
        logging.info(f"Collecting and processing data for {instrument}...")
        raw_data = fetch_historical_data(instrument, START_DATE, END_DATE, GRANULARITY)
        if raw_data.empty:
            logging.warning(f"No data found for {instrument}. Skipping...")
            continue

        processed_data = preprocess_data(raw_data)
        file_path = os.path.join(DATA_DIR, f"{instrument}_data.csv")
        processed_data.to_csv(file_path, index=False)
        logging.info(f"Processed data saved to {file_path}")

if __name__ == "__main__":
    collect_and_save_data()