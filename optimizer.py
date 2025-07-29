# optimizer.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_weights(price_data, mode="min_corr"):
    returns = price_data.pct_change().dropna()
    tickers = returns.columns
    n = len(tickers)

    cov_matrix = returns.cov().values
    std_devs = np.std(returns.values, axis=0)

    def risk_parity_objective(weights):
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        risk_contrib = weights * (cov_matrix @ weights)
        target = np.mean(risk_contrib)
        return np.sum((risk_contrib - target)**2)

    def min_corr_objective(weights):
        corr_matrix = np.corrcoef(returns.values.T)
        upper_tri = corr_matrix[np.triu_indices(n, k=1)]
        return np.dot(weights, np.dot(corr_matrix, weights))

    def constraint_sum(weights):
        return np.sum(weights) - 1

    bounds = [(0, 1) for _ in range(n)]
    init_guess = np.array([1/n]*n)

    if mode == "risk_parity":
        result = minimize(
            risk_parity_objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': constraint_sum},
            options={"disp": False}
        )
    else:
        result = minimize(
            min_corr_objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': constraint_sum},
            options={"disp": False}
        )

    if not result.success:
        raise RuntimeError("Optimization failed")

    weights = pd.Series(result.x, index=tickers)
    return weights / weights.sum()


