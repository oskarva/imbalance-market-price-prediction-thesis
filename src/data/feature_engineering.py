import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic time features derived from the DataFrame index.
    Assumes that the DataFrame index is a pandas DatetimeIndex.
    """
    df = df.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    return df

def add_lag_features(df: pd.DataFrame, columns: list, n_lags: int, impute: bool = False) -> pd.DataFrame:
    """
    Create lag features for the specified columns. If impute is True, 
    impute the missing values in the first n_lags rows; otherwise, drop them.
    """
    df = df.copy()
    for col in columns:
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    if impute:
        # Example imputation: forward fill then backfill if needed.
        df = df.fillna(method="ffill").fillna(method="bfill")
    else:
        # Drop rows with any NaN values introduced by shifting.
        df = df.dropna()
    
    return df

def add_lead_features(df: pd.DataFrame, columns: list, n_leads: int) -> pd.DataFrame:
    """
    For each column in `columns`, create lead features from lead 1 to n_leads.
    This is used for direct multi-step forecasting targets.
    """
    df = df.copy()
    for col in columns:
        for lead in range(1, n_leads + 1):
            df[f"{col}_lead{lead}"] = df[col].shift(-lead)
    df = df.dropna()
    return df


def prepare_features(df: pd.DataFrame, target_columns: list, n_lags: int, include_time: bool = True) -> pd.DataFrame:
    """
    Prepares the feature DataFrame by optionally adding time features and 
    generating lagged versions of the target variables.
    
    Parameters:
      - df: Cleaned DataFrame (with a DatetimeIndex).
      - target_columns: List of columns (usually including y and any additional autoregressive features)
        to create lagged features for.
      - n_lags: Number of lagged timesteps to include.
      - include_time: Whether to add basic time features.
    
    Returns:
      A new DataFrame that includes the original features, optional time features, and lagged target features.
    """
    if include_time:
        df = add_time_features(df)
    df = add_lag_features(df, target_columns, n_lags)
    return df
