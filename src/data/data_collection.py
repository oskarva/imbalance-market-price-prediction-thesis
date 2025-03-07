import volue_insight_timeseries 
import pandas as pd
from .feature_engineering import *
def get_data(X_curve_names, y_curve_names, session, start_date, end_date,
             add_time=False, add_lag=True, add_rolling=False, lag_value=32):
    """
    Get data for iterative forecasting model training.
    
    Returns feature matrix X with lagged values and target vector y.
    """
    # Get data for all curves
    combined_curves = X_curve_names + y_curve_names
    cleaned_df = _get_data(combined_curves, y_curve_names, session, start_date, end_date)
    
    # Make a copy for feature engineering
    df = cleaned_df.copy()
    
    # Add time features if requested
    if add_time:
        df = add_time_features(df)
    
    # Create lag features for all variables
    if add_lag:
        for col in combined_curves:
            if col in df.columns:
                df[f"{col}_lag_{lag_value}"] = df[col].shift(lag_value)
    
    # Add rolling features if requested
    if add_rolling:
        # Apply rolling features to lag columns
        lag_cols = [col for col in df.columns if f"_lag_{lag_value}" in col]
        if lag_cols:
            rolling_df = create_rolling_features(df[lag_cols], columns=lag_cols)
            # Add rolling features back to main dataframe
            for col in rolling_df.columns:
                if col not in lag_cols:  # Only add new rolling feature columns
                    df[col] = rolling_df[col]
    
    # Handle missing values (from lag creation)
    df = df.dropna()
    
    # Define feature columns for X: only lag and engineered features
    lag_cols = [col for col in df.columns if f"_lag_{lag_value}" in col]
    time_cols = [] if not add_time else [col for col in df.columns 
                                       if col not in combined_curves and 
                                       not col.endswith(f"_lag_{lag_value}") and
                                       not "roll" in col]
    rolling_cols = [] if not add_rolling else [col for col in df.columns if "roll" in col]
    
    # Combine all feature columns for X
    X_cols = lag_cols + time_cols + rolling_cols
    
    # Extract X and y
    X_df = df[X_cols]
    y_df = df[y_curve_names]
    
    # Convert to numpy arrays
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    
    # Debug info
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature columns: {X_df.columns.tolist()}")
    print(f"Target columns: {y_df.columns.tolist()}")
    
    return X, y, X_df.columns.tolist(), y_df.columns.tolist()

def _get_data(curve_names: list, target_columns: list, 
              session: volue_insight_timeseries.Session,  
              start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Retrieve data for all curves, perform resampling, drop columns (except targets)
    and drop rows with missing values. Returns a cleaned DataFrame.
    """
    pandas_series = {}
    for curve_name in curve_names:
        curve = session.get_curve(name=curve_name)
        ts = curve.get_data(data_from=start_date, data_to=end_date)
        s = ts.to_pandas()
        pandas_series[curve_name] = s

    # Upsample series with 1-hour frequency to 15-minute intervals
    for col in pandas_series:
        if " h " in col:
            pandas_series[col] = pandas_series[col].resample('15min').ffill() 

    # Build combined DataFrame with the resampled data:
    combined_df = pd.DataFrame(pandas_series)

    # Define threshold for non-target columns
    threshold = 0.4
    # Consider dropping columns that are not targets
    non_target_columns = [col for col in combined_df.columns if col not in target_columns]
    cols_to_drop = combined_df[non_target_columns].columns[
        combined_df[non_target_columns].isna().mean() > threshold
    ]
    
    # Drop only non-target columns
    combined_df = combined_df.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")

    # Drop rows with any NaN values
    cleaned_df = combined_df.dropna()

    return cleaned_df
