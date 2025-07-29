# correlation_engine.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix from price data
def compute_correlation_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    returns = price_df.pct_change().dropna()
    return returns.corr()

# Extract avg, max, min correlation and their asset pairs
def get_correlation_stats(corr: pd.DataFrame) -> dict:
    mask = ~np.eye(len(corr), dtype=bool)
    stacked = corr.where(mask).stack()
    return {
        "avg": stacked.mean(),
        "max_val": stacked.max(),
        "max_pair": stacked.idxmax(),
        "min_val": stacked.min(),
        "min_pair": stacked.idxmin()
    }

# Generate heatmap from correlation matrix
def render_heatmap(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar=True,
        fmt=".2f",
        annot_kws={"size": 9}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    return fig