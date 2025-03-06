import volue_insight_timeseries 
import pandas as pd
from .feature_engineering import *

def get_data(X_curve_names, y_curve_names, session, start_date, end_date,
             add_time=False, add_lag=False, add_rolling=False):
    # Get data from both X curves and y curves
    combined_curves = X_curve_names + y_curve_names
    cleaned_df = _get_data(combined_curves, y_curve_names, session, start_date, end_date)
    
    # Define columns
    X_columns = [col for col in X_curve_names if col in cleaned_df.columns]
    y_columns = [col for col in y_curve_names if col in cleaned_df.columns]
    
    # For future predictions, we need a copy of the complete dataset with all columns
    full_df = cleaned_df.copy()
    
    # Extract X from the DataFrame for feature engineering
    X_df = cleaned_df[X_columns].copy()
    
    # Apply transformations to X_df
    if add_time:
        X_df = add_time_features(X_df)
    
    # KEY CHANGE: Create lag features for BOTH X variables AND target variables
    if add_lag:
        # Create lag features for X columns
        X_df = create_lag_features(X_df, columns=X_columns, lags=[32])
        
        # IMPORTANT: Add lagged target variables to X_df
        for y_col in y_columns:
            lagged_target = create_lag_features(
                full_df[[y_col]], columns=[y_col], lags=[32]
            )
            # Rename to avoid confusion with the actual target
            lagged_target.columns = [f'{y_col}_target_lag_{lag}' for lag in [32]]
            
            # Join to X_df
            X_df = X_df.join(lagged_target)
    
    if add_rolling:
        X_df = create_rolling_features(X_df, columns=X_columns)
    
    if add_lag:
        X_df = impute_missing_values(X_df, method="drop")
    
    # Get valid indices after transformations
    valid_indices = X_df.index
    
    # Extract y using only the valid indices
    y_df = cleaned_df.loc[valid_indices, y_columns]
    
    # Convert to numpy arrays
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    
    # Make sure they match
    assert len(X) == len(y), f"X and y lengths don't match: {len(X)} vs {len(y)}"
    
    return X, y, list(X_df.columns), list(y_df.columns)


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
