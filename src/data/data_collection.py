import volue_insight_timeseries 
import os
import pandas as pd
import numpy as np

session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

basic_curves = [
"pri de imb up mfrr €/mwh cet min15 a",
"pri de imb down mfrr €/mwh cet min15 a",
"vol de imb sys mw cet min15 a",
"vol de imb up mw cet min15 a",
"vol de imb down mw cet min15 a",
"con de intraday mwh/h cet min15 a",
"con de heating % cet min15 s",
"con de cooling % cet min15 s",
"pro de tot mwh/h cet min15 a",
"vol de cap imb up mfrr mw cet h a",
"vol de cap imb down mfrr mw cet h a",
"vol de cap imb up afrr mw cet h a",
"vol de cap imb down afrr mw cet h a",
"pri de spot €/mwh cet h a",
"rdl de mwh/h cet min15 a",
"pri de imb stlmt €/mwh cet min15 a",
] 

exchange_curves = [
    "exc de>se4 com mw cet h a",
    "exc de>dk1 com mw cet h a",
    "exc de>dk2 com mw cet h a",
    "exc de>nl com mw cet h a",
    "exc de>fr com mw cet h a",
    "exc de>ch com mw cet h a",
    "exc de>at com mw cet h a",
    "exc de>cz com mw cet h a",
    "exc de>pl com mw cet h a",
    "exc de>se com mw cet h a",
    "exc de>no2 com mw cet h a",
    "exc de>be com mw cet h a",
    "exc de>no com mw cet h a",
    "exc de>dk com mw cet h a"
]

combined = basic_curves + exchange_curves
start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()


def get_data(X_curve_names: list, y_curve_names: list, 
             session: volue_insight_timeseries.Session,  
             start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple:
    """
    Returns a tuple (X, y) as NumPy ndarrays.
    X and y are obtained from the cleaned data based on the provided curve names.
    """
    combined_curves = X_curve_names + y_curve_names
    cleaned_df = _get_data(combined_curves, y_curve_names, session, start_date, end_date)
    
    # Optionally, add time features from the index. For example:
    cleaned_df = cleaned_df.copy()  # work on a copy to avoid SettingWithCopy warnings
    cleaned_df["year"] = cleaned_df.index.year
    cleaned_df["month"] = cleaned_df.index.month
    cleaned_df["day"] = cleaned_df.index.day
    cleaned_df["hour"] = cleaned_df.index.hour
    cleaned_df["minute"] = cleaned_df.index.minute

    # If you want the time features as part of X, append them:
    # (Otherwise, you can remove them from X by not including the names below.)
    time_features = ["year", "month", "day", "hour", "minute"]

    # Define X and y columns. For X, you can decide whether to include the time features.
    X_columns = [col for col in X_curve_names if col in cleaned_df.columns] + time_features
    y_columns = [col for col in y_curve_names if col in cleaned_df.columns]

    # Convert to numpy arrays.
    # (The rows are aligned since they come from the same cleaned_df.)
    X = cleaned_df[X_columns].to_numpy()
    y = cleaned_df[y_columns].to_numpy()

    return X, y


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
    
    # Drop only from non-target columns
    combined_df = combined_df.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")

    # Drop rows with any NaN values
    cleaned_df = combined_df.dropna()

    return cleaned_df


if __name__ == "__main__":
    # Example usage: here we choose some curves for X and target curves for y
    X, y = get_data(combined, 
                    ["pri de imb up afrr €/mwh cet min15 a", "pri de imb down afrr €/mwh cet min15 a"],
                    session, 
                    start_date, 
                    end_date)
    
    # Now X and y are numpy arrays ready for ML algorithms.
    print("X shape:", X.shape)
    print("y shape:", y.shape)