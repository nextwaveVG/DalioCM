# data_loader.py
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")

# Fetch price data for one ticker
def fetch_price_data(ticker: str, days: int) -> pd.DataFrame:
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?serietype=line&timeseries={days}&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        prices = data.get("historical", [])
        if not prices:
            return None
        df = pd.DataFrame(prices)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df[["close"]].rename(columns={"close": ticker})
    except:
        return None

# Fetch and merge all tickers
def load_price_data(ticker_list: list, days: int) -> pd.DataFrame:
    all_data = []
    for ticker in ticker_list:
        df = fetch_price_data(ticker, days)
        if df is not None:
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, axis=1).dropna()
    return None
