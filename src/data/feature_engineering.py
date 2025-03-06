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
import pandas as pd
import numpy as np

def create_lag_features(df, columns, lags):
    """
    Create lag features for specified columns.
    
    Args:
        df (pd.DataFrame): Dataframe containing the time series data.
        columns (list of str): Column names for which lag features are to be created.
        lags (list of int): Lag intervals (in number of rows) to shift the data.
        
    Returns:
        pd.DataFrame: Dataframe with new lag features appended.
    """
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def create_rolling_features(df, columns, windows, agg_funcs=['mean']):
    """
    Create rolling aggregate features for specified columns.
    
    Args:
        df (pd.DataFrame): Dataframe containing the time series data.
        columns (list of str): Column names for which rolling features are to be computed.
        windows (list of int): List of window sizes (in number of rows) for rolling calculations.
        agg_funcs (list of str): List of aggregate functions to apply (e.g., 'mean', 'sum', 'std').
        
    Returns:
        pd.DataFrame: Dataframe with new rolling feature columns appended.
    """
    df = df.copy()
    for col in columns:
        for window in windows:
            for func in agg_funcs:
                df[f'{col}_roll_{window}_{func}'] = df[col].rolling(window=window, min_periods=1).agg(func)
    return df

def add_calendar_features(df, datetime_col):
    """
    Extract and add calendar features (hour, day of week, month, etc.) from a datetime column.
    
    Args:
        df (pd.DataFrame): Dataframe containing a datetime column.
        datetime_col (str): Column name of the datetime field.
        
    Returns:
        pd.DataFrame: Dataframe with additional calendar features.
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek  # Monday=0, Sunday=6
    df['day'] = df[datetime_col].dt.day
    df['month'] = df[datetime_col].dt.month
    # Week of year might be useful to capture weekly seasonality
    df['week_of_year'] = df[datetime_col].dt.isocalendar().week.astype(int)
    return df

def impute_missing_values(df, method='ffill'):
    """
    Handle missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with missing values.
        method (str): Imputation method: 
                      - 'ffill': forward-fill,
                      - 'bfill': backward-fill,
                      - 'interpolate': linear interpolation,
                      - 'drop': drop rows with missing values.
                      
    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    df = df.copy()
    if method == 'ffill':
        df = df.fillna(method='ffill')
    elif method == 'bfill':
        df = df.fillna(method='bfill')
    elif method == 'interpolate':
        df = df.interpolate()
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError("Method must be one of: 'ffill', 'bfill', 'interpolate', 'drop'")
    return df

# Example usage:
if __name__ == "__main__":
    # For demonstration, create a dummy time series dataframe.
    date_rng = pd.date_range(start='2020-01-01', end='2020-01-02', freq='15T')
    df = pd.DataFrame(date_rng, columns=['timestamp'])
    df['price'] = np.random.randn(len(date_rng))  # Example target variable
    df['volume'] = np.random.randint(0, 100, len(date_rng))  # Example exogenous variable

    # Add calendar features based on the 'timestamp' column.
    df = add_calendar_features(df, 'timestamp')
    
    # Create lag features for 'price' and 'volume'.
    # Here, we create lags of 1, 2, and 3 time steps (adjust based on your forecasting horizon).
    df = create_lag_features(df, columns=['price', 'volume'], lags=[1, 2, 3])
    
    # Create rolling features (e.g., rolling mean and sum) for 'price'
    # Using window sizes of 4 and 8 time steps (e.g., 1 hour or 2 hours for 15-min data).
    df = create_rolling_features(df, columns=['price'], windows=[4, 8], agg_funcs=['mean', 'sum'])
    
    # Handle missing values (forward fill in this example)
    df = impute_missing_values(df, method='ffill')
    
    # Show the resulting dataframe with engineered features
    print(df.head(10))