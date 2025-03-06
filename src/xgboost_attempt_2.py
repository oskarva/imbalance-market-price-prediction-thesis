import os
import math
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Import data extraction methods and curve definitions
from data.data_collection import get_data
from data.feature_engineering import create_lag_features, create_rolling_features, impute_missing_values
from data.curves import curve_collections
import volue_insight_timeseries

# ---------------------------
# Setup Output Folder
# ---------------------------
output_dir = os.path.join("results", "xgboost_rolling_predictions_tuned_full_custom_shift")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Parameters and Data Loading
# ---------------------------
start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()

# Define the curves:
# X_curve_names: additional independent curves.
# target_curve: the target variable.
X_curve_names = curve_collections["de"]["X"]    # additional curves
target_curve = curve_collections["de"]["mfrr"][0]  # e.g. "mfrr_up_price" or similar

session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

X, y, X_columns, y_columns = get_data(X_curve_names, target_curve,
                                      session,
                                      start_date, end_date,
                                      add_time=False, 
                                      add_lag=True,
                                      add_rolling=False,
                                      )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)