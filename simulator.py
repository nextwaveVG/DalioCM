# simulator.py
import pandas as pd
import numpy as np

# Monte Carlo simulation for future portfolio values
def monte_carlo_simulation(price_df: pd.DataFrame, weights: pd.Series, n_simulations=1000, n_days=252):
    returns = price_df.pct_change().dropna()
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    weights = weights.values.reshape(-1, 1)

    simulations = np.zeros((n_simulations, n_days))
    for i in range(n_simulations):
        simulated_daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=n_days)
        portfolio_returns = simulated_daily_returns @ weights
        simulations[i] = np.cumprod(1 + portfolio_returns.flatten())

    return pd.DataFrame(simulations)

# Historical backtest for selected weights
def backtest_portfolio(price_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    returns = price_df.pct_change().dropna()
    weighted_returns = returns @ weights
    cumulative = (1 + weighted_returns).cumprod()
    return cumulative