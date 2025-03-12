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
    Get forecast for the next 32 15-minute timesteps.
    """

    # Get last date in training set
    last_date = X_train.index[-1]
    # Get forecast start date
    forecast_start_date = last_date + pd.Timedelta(minutes=15)
    # Get forecast end date
    forecast_end_date = forecast_start_date + pd.Timedelta(minutes=15*32)
    # Create forecast dates
    forecast_dates = pd.date_range(forecast_start_date, forecast_end_date, freq='15min')
    # Create forecast DataFrame
    X_forecast = pd.DataFrame(index=forecast_dates)
    # Set columns to avoid KeyError
    for orig in X_to_forecast.keys():
        X_forecast[orig] = None
    
    # Get forecast data
    for (orig, col) in X_to_forecast.items():

        if col is None:
            # I need to add empty rows to a copy of X_train for the rows that are to be forecasted
            # Then I need to fill the missing values in these rows with Prophet
            # Then I need to get the last 32 values of the forecasted rows
            # Then I need to add these values to X_forecast
            X_train_copy = X_train.copy()
            # Add empty rows
            X_train_copy = pd.concat([X_train_copy, pd.DataFrame(index=forecast_dates, columns=X_train_copy.columns)])

            result = fill_missing_with_prophet(df=X_train_copy.reset_index(), columns_to_fill=[orig], date_column="index")
            # get only the 32 forecast values
            result = result.set_index("index")

            X_forecast[orig] = result[orig].iloc[-32:]
        else:
            curve = session.get_curve(name=col)
            if type(curve) is InstanceCurve:
                ts = curve.get_instance(
                    issue_date=forecast_start_date,
                    data_from=forecast_start_date,
                    data_to=forecast_end_date,
                    )
            else:
                ts = curve.get_data(data_from=forecast_start_date, data_to=forecast_end_date)
            s = ts.to_pandas()
            X_forecast[orig] = s

            # Upsample series with 1-hour frequency to 15-minute intervals
            if " h " in col:
                X_forecast[orig] = X_forecast[orig].resample('15min').ffill()

    # if there is a NaN value in the forecast, fill it with the last known value
    X_forecast = X_forecast.fillna(method='ffill')
    
    return X_forecast
