import volue_insight_timeseries 
import os
import pandas as pd
import matplotlib.pyplot as plt

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

#TODO: Add net transfer capacity curves?

combined = basic_curves + exchange_curves
start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()
def get_data(X_curve_names: list, y_curve_names: list, session: volue_insight_timeseries.Session,  start_date: pd.Timestamp, end_date: pd.Timestamp):
    combined_curves = X_curve_names + y_curve_names
    combined_data = _get_data(combined_curves, session, start_date, end_date)

    # Now, split the cleaned combined_data into X and y:
    X = {col: combined_data[col] for col in X_curve_names if col in combined_data}
    y = {col: combined_data[col] for col in y_curve_names if col in combined_data}

    return X, y

def _get_data(curve_names: list, session: volue_insight_timeseries.Session,  start_date: pd.Timestamp, end_date: pd.Timestamp):

    pandas_series = {}
    for curve_name in curve_names:
        curve = session.get_curve(name=curve_name)
        ts = curve.get_data(data_from=start_date, data_to=end_date)
        s = ts.to_pandas()
        pandas_series[curve_name] = s

    # Combine all series into a single DataFrame
    combined_df = pd.DataFrame(pandas_series)

    # Drop columns with more than a certain threshold of NaN values
    # For example, drop columns with more than 50% NaN values
    threshold = 0.4
    cols_to_drop = combined_df.columns[combined_df.isna().mean() > threshold]
    combined_df = combined_df.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")

    # Drop rows with any NaN values
    cleaned_df = combined_df.dropna()

    # Convert the cleaned DataFrame back to a dictionary of series
    pandas_series = {col: cleaned_df[col] for col in cleaned_df}

    # Upsample series with 1-hour frequency to 15-minute intervals
    for col in pandas_series:
        if " h " in col:
            pandas_series[col] = pandas_series[col].resample('15T').ffill() 

    # Reset index and convert datetime to numerical features
    for col in pandas_series:
        df = pandas_series[col].reset_index()
        df['year'] = df['index'].dt.year
        df['month'] = df['index'].dt.month
        df['day'] = df['index'].dt.day
        df['hour'] = df['index'].dt.hour
        df['minute'] = df['index'].dt.minute
        df.drop(columns=['index'], inplace=True)
        pandas_series[col] = df
    
    return pandas_series

if __name__ == "__main__":
    data = get_data(combined, 
                    ["pri de imb up afrr €/mwh cet min15 a", "pri de imb down afrr €/mwh cet min15 a"],
                     session, 
                     start_date, 
                     end_date)


