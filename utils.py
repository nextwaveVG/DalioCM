# utils.py

def format_percentage_series(series):
    return series.apply(lambda x: f"{x:.2%}")

def format_decimal_series(series):
    return series.apply(lambda x: f"{x:.2f}")

def debug_log(message):
    print(f"[DEBUG] {message}")

# Optionally used later

def normalize_weights(weights):
    import numpy as np
    total = np.sum(weights)
    return weights / total if total != 0 else weights
