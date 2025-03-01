import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import volue_insight_timeseries 
from data.curves import curve_collections
from data.data_collection import get_data

# Import (or define) your naive model.
from models.naive import Naive_Last_Known_Activation_Price

# ---------------------------
# Load Data and Create Initial Train/Test Split
# ---------------------------
start_date = pd.Timestamp("2024-06-06")
end_date = pd.Timestamp.today()
session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

# Define the curve names (using your provided collections)
X_curve_names = curve_collections["de"]["X"]
y_curve_names = [curve_collections["de"]["mfrr"][0]]

# Get the data (assumed to return a DataFrame or similar structure)
X, y, X_col, _ = get_data(X_curve_names, y_curve_names, session, start_date, end_date)

# If y is a numpy array, convert it to a pandas Series with a datetime index.
if isinstance(y, np.ndarray):
    # Assuming one row per hour
    time_index = pd.date_range(start=start_date, periods=len(y), freq='H')
    y = pd.Series(y.flatten(), index=time_index, name=curve_collections["de"]["mfrr"][0])
else:
    # If it's already a DataFrame/Series, sort by index.
    y = y.sort_index()

# Instead of a random train/test split (which is not ideal for time series),
# we select an initial training period and then use the remaining data for rolling forecasts.
# Here we use the first 10% of the data as initial training.
split_idx = int(len(y) * 0.1)
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# ---------------------------
# Rolling Forecast Settings
# ---------------------------
rolling_window = 8  # Forecast 8 hours ahead in each rolling iteration.

# Lists to store the predictions and corresponding true values.
predictions = []
true_values = []

# This variable will hold our expanding training set.
current_train_y = y_train.copy()

# Loop over the test set in steps of the rolling window.
# In each iteration:
# 1. Train the model using the current training set.
# 2. Predict the next 8 hours (the naive model always predicts the last seen value).
# 3. Record the predictions and the actual values.
# 4. Append the just-forecasted data to the training set.
for start in range(0, len(y_test), rolling_window):
    end = start + rolling_window
    test_window = y_test.iloc[start:end]
    if test_window.empty:
        break

    # Create and train the naive model on the current training set.
    model = Naive_Last_Known_Activation_Price()
    model.train(current_train_y.to_frame(), column_name=curve_collections["de"]["mfrr"][0])
    
    # Predict for the next rolling_window hours.
    # Since this is a naive model, it always returns the same value.
    pred = model.predict()
    # Create an array filled with the prediction.
    pred_array = np.full(len(test_window), pred)
    
    # Collect the predictions and the actual values.
    predictions.extend(pred_array)
    true_values.extend(test_window.values)
    
    # Extend the training data with the test window (simulating a rolling forecast update).
    current_train_y = pd.concat([current_train_y, test_window])

# ---------------------------
# Final Evaluation Metrics
# ---------------------------
final_r2 = r2_score(true_values, predictions)
final_rmse = np.sqrt(mean_squared_error(true_values, predictions))

print("Final R2:", final_r2)
print("Final RMSE:", final_rmse)

import matplotlib.pyplot as plt
import pandas as pd

# Create a predictions Series using the same index as y_test.
pred_series = pd.Series(predictions, index=y_test.index[:len(predictions)])
true_series = y_test.iloc[:len(predictions)]

# ---------------------------
# Plot 1: True vs Predicted Prices
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(true_series.index, true_series, label='True Prices', color='blue')
plt.plot(pred_series.index, pred_series, label='Predicted Prices', color='red', linestyle='--')
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("True vs Predicted Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Plot 2: Residual Histogram
# ---------------------------
errors = true_series.values - pred_series.values
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Forecast Error (True - Predicted)")
plt.ylabel("Frequency")
plt.title("Histogram of Forecast Errors")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Plot 3: Absolute Error Over Time
# ---------------------------
absolute_errors = np.abs(errors)
plt.figure(figsize=(12, 6))
plt.plot(true_series.index, absolute_errors, marker='o', linestyle='-', color='purple')
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.title("Absolute Forecast Error Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()