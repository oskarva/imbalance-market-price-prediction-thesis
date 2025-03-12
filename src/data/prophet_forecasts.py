import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import timedelta

def fill_missing_with_prophet(df, columns_to_fill, date_column='date'):
    """
    Fill missing values in a dataframe for specific columns using Prophet forecasting.
    Optimized for 15-minute interval data with proper seasonality patterns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe with timeseries data
    columns_to_fill : list
        List of column names that need to be filled with forecasts
    date_column : str, default='date'
        Name of the column containing datetime information
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with missing values filled with Prophet forecasts
    """
    # Make a copy of the input dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Ensure the date column is in datetime format
    result_df[date_column] = pd.to_datetime(result_df[date_column])
    
    # Handle timezone issues
    if result_df[date_column].dt.tz is not None:
        original_tz = result_df[date_column].dt.tz
        # Convert to timezone-naive by converting to UTC and then removing timezone info
        result_df[date_column] = result_df[date_column].dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        original_tz = None
    
    # Set index and sort
    result_df = result_df.set_index(date_column)
    result_df = result_df.sort_index()
    
    # Process each column that needs filling
    for column in columns_to_fill:
        print(f"Processing column: {column}")
        
        # Find ranges of missing data
        is_missing = result_df[column].isna()
        
        # If no missing values, skip this column
        if not is_missing.any():
            print(f"No missing values in column {column}, skipping.")
            continue
        
        # Identify continuous missing ranges
        missing_ranges = []
        missing_start = None
        
        for idx, missing in zip(result_df.index, is_missing):
            if missing and missing_start is None:
                missing_start = idx
            elif not missing and missing_start is not None:
                missing_ranges.append((missing_start, idx - timedelta(minutes=15)))  # Adjusted for 15-min intervals
                missing_start = None
                
        # Check if the last range is still open
        if missing_start is not None:
            missing_ranges.append((missing_start, result_df.index[-1]))
        
        # Process each missing range
        for start_date, end_date in missing_ranges:
            print(f"Filling missing data from {start_date} to {end_date}")
            
            # Get training data before the missing range
            # For 15-min data, try to get at least a week of data
            train_data = result_df.loc[:start_date - timedelta(minutes=15), column].dropna()
            
            # Skip if not enough training data (at least 96*7 points = 1 week)
            min_training_points = 96 * 7  # 96 points per day × 7 days
            if len(train_data) < min_training_points:
                print(f"Not enough training data before {start_date}, trying with available data.")
                # If less than ideal but still have some data, proceed with caution
                if len(train_data) < 96:  # At least one day of data
                    print(f"Too little data (<1 day) to train model, using last known value instead.")
                    # Fill with last known value
                    last_value = train_data.iloc[-1] if not train_data.empty else 0
                    missing_mask = (result_df.index >= start_date) & (result_df.index <= end_date)
                    result_df.loc[missing_mask, column] = last_value
                    continue
            
            # Prepare data for Prophet - ensure at least daily seasonality
            train_df = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data.values
            })
            
            # Configure Prophet model for 15-minute data
            model = Prophet(
                daily_seasonality=True,  # Always include daily patterns
                weekly_seasonality=True,  # Weekly patterns are often important
                yearly_seasonality=True if len(train_data) > 96*365 else False,  # Yearly if enough data
                changepoint_prior_scale=0.05,  # Slightly more flexible than default
                seasonality_prior_scale=10.0,  # Emphasize seasonality 
                changepoint_range=0.95  # Allow changes up to 95% of the training data
            )
            
            # Add hourly seasonality (crucial for 15-min data)
            model.add_seasonality(
                name='hourly',
                period=24/24,  # 1 hour period in days
                fourier_order=8  # Higher order for more flexibility in the curve
            )
            
            # Add quarter-hourly seasonality if we have enough data
            if len(train_data) > 96*14:  # Two weeks of data
                model.add_seasonality(
                    name='quarter_hourly',
                    period=1/96,  # 15 minutes in days
                    fourier_order=3  # Lower order - simpler pattern
                )
            
            # Fit the model
            model.fit(train_df)
            
            # Create future dataframe including the missing range WITH 15-MIN FREQUENCY
            future_dates = pd.date_range(
                start=start_date,
                end=end_date,
                freq='15min'  # Crucial: use 15-min frequency to match data
            )
            
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Generate forecasts
            forecast = model.predict(future_df)
            
            # Fill the missing values in the result dataframe
            for forecast_date, yhat in zip(forecast['ds'], forecast['yhat']):
                if forecast_date in result_df.index:
                    result_df.loc[forecast_date, column] = yhat
    
    # Reset index to return the date as a column
    result_df = result_df.reset_index()
    
    # Convert back to original timezone if necessary
    if original_tz is not None:
        result_df[date_column] = result_df[date_column].dt.tz_localize('UTC').dt.tz_convert(original_tz)
    
    return result_df

# Example usage
if __name__ == "__main__":
    # Sample data with missing values
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    data = {
        'date': dates,
        'feature1': np.random.normal(10, 2, 60),
        'feature2': np.random.normal(20, 5, 60)
    }
    
    df = pd.DataFrame(data)
    
    # Create missing data ranges (t to t-32 for some features)
    # Let's say feature2 is missing from day 30 to day 45
    df.loc[30:45, 'feature2'] = np.nan
    
    # Fill missing values using Prophet
    filled_df = fill_missing_with_prophet(df, columns_to_fill=['feature2'])
    
    print(filled_df[['date', 'feature2']][25:50])  # Show filled values