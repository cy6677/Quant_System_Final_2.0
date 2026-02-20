import pandas as pd

def validate_missing(df: pd.DataFrame, max_missing_pct=0.2):
    missing_pct = df.isna().mean()
    bad = missing_pct[missing_pct > max_missing_pct]
    return bad

def validate_spikes(df: pd.DataFrame, threshold=0.5):
    returns = df.pct_change()
    spikes = returns.abs() > threshold
    return spikes.sum().sum()

def validate_spikes_by_ticker(df: pd.DataFrame, threshold=0.5):
    returns = df.pct_change()
    spikes = returns.abs() > threshold
    return spikes.sum()