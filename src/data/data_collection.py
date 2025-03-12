import volue_insight_timeseries 
from volue_insight_timeseries.curves import InstanceCurve
import pandas as pd
from .feature_engineering import *
from .prophet_forecasts import fill_missing_with_prophet

def get_data(X_curve_names, y_curve_names, session, start_date, end_date, X_to_forecast,
             add_time=False, add_lag=False, add_rolling=False, include_y_in_X=False, lag_value=32, initial_training_set_size=0.8):
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
    
    if X_cols == []:
        X_cols = X_curve_names.copy()
    
    if include_y_in_X:
        X_cols += y_curve_names

    # Extract X and y
    X_df = df[X_cols]
    y_df = df[y_curve_names]

    n_rounds = create_cross_validation_sets_and_save(X_df, y_df, initial_training_set_size, X_to_forecast, session, crossval_horizon=32)
    
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

def create_cross_validation_sets_and_save(X_df, y_df, initial_training_set_size, X_to_forecast, session, crossval_horizon):
    """
    Create cross-validation sets and save them to disk.
    """
    # Calculate number of crossvalidation rounds
    n_rounds = int((1 - initial_training_set_size) * len(X_df) / crossval_horizon)

    # Create cross-validation sets
    for i in range(n_rounds):
        X_train = X_df.iloc[:int(initial_training_set_size * len(X_df)) + i * crossval_horizon]
        y_train = y_df.iloc[:int(initial_training_set_size * len(X_df)) + i * crossval_horizon]
        y_test = y_df.iloc[int(initial_training_set_size * len(X_df)) + i * crossval_horizon:int(initial_training_set_size * len(X_df)) + (i + 1) * crossval_horizon]

        #Get dates for forecast (I need to not include the dates where there have been NaN values in the actuals)
        dates = y_test.index

        X_test = get_forecast(dates, X_train, X_to_forecast, session)
        #TODO: Accounter jeg her for at noen kolonner kan være droppet? Tror ikke jeg oppdaterer listen over X_kollonner (som jeg ikke bruker her enda) basert på droppede kolonner.
        
        X_train.to_csv(f"./src/data/csv/X_train_{i}.csv", index=False)
        y_train.to_csv(f"./src/data/csv/y_train_{i}.csv", index=False)
        X_test.to_csv(f"./src/data/csv/X_test_{i}.csv", index=False)
        y_test.to_csv(f"./src/data/csv/y_test_{i}.csv", index=False)
    
    return n_rounds

def get_forecast(dates, X_train, X_to_forecast, session):
    """
    Get forecast for the next 32 15-minute timesteps, with proper handling of 15-minute intervals
    and special attention to the first forecast value.
    """
    # Get last date in training set
    last_date = X_train.index[-1]
    
    # Get forecast start date (next 15-min interval)
    forecast_start_date = last_date + pd.Timedelta(minutes=15)
    
    # Get forecast end date (32 intervals ahead)
    forecast_end_date = forecast_start_date + pd.Timedelta(minutes=15*31)  # 31 intervals after start = 32 total
    
    # Create forecast dates with explicit 15-min frequency
    forecast_dates = pd.date_range(forecast_start_date, forecast_end_date, freq='15min')
    
    # Verify we have exactly 32 forecast points
    if len(forecast_dates) != 32:
        print(f"Warning: Expected 32 forecast points, but generated {len(forecast_dates)}")
    
    # Create forecast DataFrame with all columns from X_train
    X_forecast = pd.DataFrame(index=forecast_dates, columns=X_train.columns)
    
    # Create a tracking set for columns with data issues
    problem_columns = set()
    
    # Process each column in X_to_forecast 
    for (orig, col) in X_to_forecast.items():
        if col is None:
            # Need to forecast this column with Prophet
            print(f"Forecasting column {orig} with Prophet")
            
            # Prepare data for Prophet
            X_train_copy = X_train.copy()
            
            # Add empty rows for the forecast period
            forecast_df = pd.DataFrame(index=forecast_dates, columns=X_train_copy.columns)
            X_train_copy = pd.concat([X_train_copy, forecast_df])
            
            # Reset index for Prophet
            df_for_prophet = X_train_copy.reset_index()
            
            # Use updated Prophet function to fill missing values
            result = fill_missing_with_prophet(
                df=df_for_prophet, 
                columns_to_fill=[orig], 
                date_column="index"
            )
            
            # Set index back and get forecast values
            result = result.set_index("index")
            
            # Extract only the 32 forecast values we need
            X_forecast[orig] = result[orig].loc[forecast_dates]
            
            # Check if the first value would need to be filled
            if pd.isna(X_forecast[orig].iloc[0]):
                problem_columns.add(orig)
            
        else:
            # Get data from an external forecast curve
            print(f"Fetching external forecast for {orig} from curve {col}")
            curve = session.get_curve(name=col)
            
            if type(curve) is volue_insight_timeseries.curves.InstanceCurve:
                ts = curve.get_instance(
                    issue_date=forecast_start_date,
                    data_from=forecast_start_date,
                    data_to=forecast_end_date,
                )
            else:
                ts = curve.get_data(data_from=forecast_start_date, data_to=forecast_end_date)
            
            s = ts.to_pandas()
            X_forecast[orig] = s
            
            # Upsample hourly data to 15-minute intervals if needed
            if " h " in col:
                X_forecast[orig] = X_forecast[orig].resample('15min').ffill()
            
            # Check if the first value would need to be filled
            if pd.isna(X_forecast[orig].iloc[0]):
                problem_columns.add(orig)
    
    # For columns not explicitly specified in X_to_forecast, derive forecast
    # using appropriate strategies
    missing_columns = [col for col in X_forecast.columns if col not in X_to_forecast.keys()]
    
    if missing_columns:
        print(f"Handling additional columns: {missing_columns}")
        
        for col in missing_columns:
            # Check if we should use Prophet or a simpler approach
            use_prophet = True  # Set to False if you want to use simple approach for some columns
            
            if use_prophet:
                # Use Prophet for forecasting this column
                X_train_copy = X_train.copy()
                forecast_df = pd.DataFrame(index=forecast_dates, columns=[col])
                X_train_copy = pd.concat([X_train_copy, forecast_df])
                
                df_for_prophet = X_train_copy.reset_index()
                result = fill_missing_with_prophet(
                    df=df_for_prophet, 
                    columns_to_fill=[col], 
                    date_column="index"
                )
                result = result.set_index("index")
                X_forecast[col] = result[col].loc[forecast_dates]
                
                # Check if the first value would need to be filled
                if pd.isna(X_forecast[col].iloc[0]):
                    problem_columns.add(col)
            else:
                # Simpler approach: use last known value
                X_forecast[col] = X_train[col].iloc[-1]
    
    # SPECIAL HANDLING FOR FIRST VALUES
    # First, check which columns have NaN in the first position
    nan_in_first = X_forecast.iloc[0].isna()
    columns_with_nan_first = nan_in_first[nan_in_first].index.tolist()
    
    # For each column with NaN in first position, apply a better strategy than backfill
    for col in columns_with_nan_first:
        print(f"Fixing first value for column: {col}")
        
        # Strategy 1: Use Prophet prediction if possible
        if X_train[col].notna().any():
            # Get last few values from training data to establish trend
            last_values = X_train[col].iloc[-5:].dropna()
            
            if len(last_values) >= 2:
                # Calculate simple linear trend from last points
                steps = range(len(last_values))
                coeffs = np.polyfit(steps, last_values, 1)
                
                # Project one step forward using the trend
                next_value = coeffs[0] * len(last_values) + coeffs[1]
                
                # Apply small random variation to avoid exact duplication
                variation = 0.01 * np.std(last_values) if len(last_values) > 2 else 0
                next_value += np.random.normal(scale=variation)
                
                # Ensure value is reasonable (not negative for values that shouldn't be)
                if last_values.min() >= 0 and next_value < 0:
                    next_value = 0
                
                X_forecast.iloc[0, X_forecast.columns.get_loc(col)] = next_value
                
                # Add small variation to second value if it's also NaN
                if pd.isna(X_forecast.iloc[1, X_forecast.columns.get_loc(col)]):
                    second_value = next_value + coeffs[0] + np.random.normal(scale=variation)
                    if last_values.min() >= 0 and second_value < 0:
                        second_value = 0
                    X_forecast.iloc[1, X_forecast.columns.get_loc(col)] = second_value
            else:
                # Not enough data for trend - use last value with small variation
                if len(last_values) > 0:
                    last_val = last_values.iloc[-1]
                    variation = 0.02 * abs(last_val) if last_val != 0 else 0.1
                    next_value = last_val + np.random.normal(scale=variation)
                    X_forecast.iloc[0, X_forecast.columns.get_loc(col)] = next_value
                    
                    # Also fix second value if needed
                    if pd.isna(X_forecast.iloc[1, X_forecast.columns.get_loc(col)]):
                        second_value = next_value + np.random.normal(scale=variation)
                        X_forecast.iloc[1, X_forecast.columns.get_loc(col)] = second_value
        
        # If still NaN after all this, fall back to forward fill
        if pd.isna(X_forecast.iloc[0, X_forecast.columns.get_loc(col)]):
            X_forecast[col] = X_forecast[col].fillna(method='ffill')
    
    # Now handle any remaining NaN values
    # Apply forward fill first
    X_forecast = X_forecast.fillna(method='ffill')
    
    # Use backward fill as backup
    X_forecast = X_forecast.fillna(method='bfill')
    
    # As a last resort, fill remaining NaNs with zeros
    final_nan_count = X_forecast.isna().sum().sum()
    if final_nan_count > 0:
        print(f"Warning: Still have {final_nan_count} NaN values after ffill/bfill, filling with zeros")
        X_forecast = X_forecast.fillna(0)
    
    # Verify columns previously identified with problems
    for col in problem_columns:
        if X_forecast[col].iloc[0] == X_forecast[col].iloc[1]:
            print(f"Warning: First two values still identical in column {col}")
            
            # Apply additional random variation to second value
            variation = 0.03 * abs(X_forecast[col].iloc[0]) if X_forecast[col].iloc[0] != 0 else 0.1
            X_forecast.iloc[1, X_forecast.columns.get_loc(col)] += np.random.normal(scale=variation)
    
    # Final verification
    if X_forecast.isna().sum().sum() > 0:
        print("ERROR: NaN values still present in forecast data!")
    
    return X_forecast
