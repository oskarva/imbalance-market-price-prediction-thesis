import os
import math
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import data extraction methods and curve definitions
from data.data_collection import get_data
from data.feature_engineering import create_lag_features, create_rolling_features, impute_missing_values
from data.curves import curve_collections
import volue_insight_timeseries

start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp(year=2025, month=3, day=13)

X_curve_names = curve_collections["de"]["X"] 
target_curve = curve_collections["de"]["mfrr"][0] 

session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

X, y, X_columns, y_columns, n_rounds = get_data(X_curve_names, [target_curve],
                                      session,
                                      start_date, end_date,
                                      curve_collections["de"]["X_to_forecast"],
                                      add_time=False, 
                                      add_lag=False,
                                      add_rolling=False,
                                      )

print(n_rounds)