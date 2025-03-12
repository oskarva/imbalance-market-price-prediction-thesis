import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import timedelta

def fill_missing_with_prophet(df, columns_to_fill, date_column='date'):
    """
    Fill missing values in a dataframe for specific columns using Prophet forecasting.
    
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
    
    # Ensure the date column is in datetime format and set as index
    result_df[date_column] = pd.to_datetime(result_df[date_column])
    result_df = result_df.set_index(date_column)
    
    # Sort by date to ensure time order
    result_df = result_df.sort_index()
    
    # Process each column that needs filling
    for column in columns_to_fill:
        print(f"Processing column: {column}")
        
        # Find ranges of missing data
        # Create a series indicating where values are missing
        is_missing = result_df[column].isna()
        
        # If no missing values, skip this column
        if not is_missing.any():
            print(f"No missing values in column {column}, skipping.")
            continue
        
        # Get the start and end indices of missing data blocks
        missing_ranges = []
        missing_start = None
        
        for idx, missing in zip(result_df.index, is_missing):
            if missing and missing_start is None:
                missing_start = idx
            elif not missing and missing_start is not None:
                missing_ranges.append((missing_start, idx - timedelta(days=1)))
                missing_start = None
                
        # Check if the last range is still open
        if missing_start is not None:
            missing_ranges.append((missing_start, result_df.index[-1]))
        
        # Process each missing range
        for start_date, end_date in missing_ranges:
            print(f"Filling missing data from {start_date} to {end_date}")
            
            # Get training data before the missing range
            train_data = result_df.loc[:start_date - timedelta(days=1), column].dropna()
            
            # Skip if not enough training data
            if len(train_data) < 14:  # Need at least 2 weeks of data
                print(f"Not enough training data before {start_date}, skipping this range.")
                continue
            
            # Prepare data for Prophet
            train_df = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data.values
            })
            
            # Train Prophet model
            model = Prophet(daily_seasonality=True)
            model.fit(train_df)
            
            # Create future dataframe including the missing range
            future_dates = pd.date_range(
                start=start_date,
                end=end_date,
                freq='D'
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