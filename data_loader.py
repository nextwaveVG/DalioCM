# data_loader.py
import pandas as pd
import yfinance as yf

def fetch_price_data(ticker: str, days: int) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d")
        if df.empty:
            return None
        df = df[["Close"]].rename(columns={"Close": ticker})
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch {ticker}: {e}")
        return None

def load_price_data(ticker_list: list, days: int) -> pd.DataFrame:
    all_data = []
    for ticker in ticker_list:
        df = fetch_price_data(ticker, days)
        if df is not None:
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, axis=1).dropna()
    return None

