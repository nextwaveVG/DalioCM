# risk_metrics.py
import pandas as pd
import numpy as np

# Calculate standard deviation (volatility)
def calculate_volatility(price_df: pd.DataFrame) -> pd.Series:
    returns = price_df.pct_change().dropna()
    return returns.std() * np.sqrt(252)  # Annualized std dev

# Calculate Sharpe Ratio per asset (assuming risk-free rate = 0)
def calculate_sharpe(price_df: pd.DataFrame) -> pd.Series:
    returns = price_df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = mean_returns / volatility
    return sharpe_ratio

# Calculate rolling beta vs benchmark (e.g., SPY or BTC-USD)
def calculate_rolling_beta(price_df: pd.DataFrame, benchmark: str, window: int = 30) -> pd.DataFrame:
    returns = price_df.pct_change().dropna()
    benchmark_ret = returns[benchmark]
    betas = {}
    for col in returns.columns:
        if col == benchmark:
            continue
        cov = returns[col].rolling(window).cov(benchmark_ret)
        var = benchmark_ret.rolling(window).var()
        betas[col] = cov / var
    return pd.DataFrame(betas)
