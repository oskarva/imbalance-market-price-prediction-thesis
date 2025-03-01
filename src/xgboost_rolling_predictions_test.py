import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Import data extraction methods and curve definitions
from data.data_collection import get_data
from data.curves import curve_collections
import volue_insight_timeseries

# ---------------------------
# Setup Output Folder
# ---------------------------
output_dir = os.path.join("results", "xgboost")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Parameters and Data Loading
# ---------------------------
start_date = pd.Timestamp("2024-06-06")
end_date = pd.Timestamp.today()
target_curve = curve_collections["de"]["mfrr"][0]  # target column name

# For this XGBoost example, we use only the target variable (for lag features)
# If you need exogenous features, you can include them by adding appropriate columns.
session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))
# Pass an empty list for X_curve_names if you only want the target variable
_, y_data, _, _ = get_data([], [target_curve], session, start_date, end_date)

# Convert y_data to DataFrame with a 15-minute frequency index
if isinstance(y_data, np.ndarray):
    time_index = pd.date_range(start=start_date, periods=len(y_data), freq='15min')
    df = pd.DataFrame(y_data, index=time_index, columns=[target_curve])
else:
    df = y_data.copy().sort_index()

# ---------------------------
# Feature Engineering: Create Lag (Shift) Features
# ---------------------------
def create_lag_features(df, target, num_lags):
    """
    Create lag features for the target column.
    For each lag in 1..num_lags, a new column 'target_lag{lag}' is created.
    """
    for lag in range(1, num_lags + 1):
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df

# Use a lag window of 8 (this can be tuned; you might try longer lags if needed)
num_lags = 8
df = create_lag_features(df, target_curve, num_lags)
df.dropna(inplace=True)  # drop rows with NaN lag values

# Define feature columns and target column
feature_cols = [f"{target_curve}_lag{lag}" for lag in range(1, num_lags + 1)]
target_col = target_curve

# ---------------------------
# Rolling-Origin Cross-Validation Setup
# ---------------------------
# Use the first 10% of data as the initial training set; the rest is used for iterative forecasting.
split_idx = int(len(df) * 0.1)
df_train_initial = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

# Forecast horizon: 32 timesteps (8 hours ahead)
rolling_window = 32

# Containers to store predictions and corresponding true values
all_predictions = []
all_true_values = []
prediction_timestamps = []

# This DataFrame will be updated (expanded) as new test observations become available.
current_train_df = df_train_initial.copy()

## ---------------------------
# Iterative Forecasting: Rolling-Origin (Walk-Forward) Validation
# ---------------------------
for start in range(0, len(df_test), rolling_window):
    end = start + rolling_window
    test_window = df_test.iloc[start:end].copy()
    if test_window.empty:
        break

    # Prepare training data for this fold
    X_train = current_train_df[feature_cols].values
    y_train = current_train_df[target_col].values

    # Train XGBoost model (one-step-ahead predictor)
    xgb_model = xgb.XGBRegressor(random_state=42, n_estimators=100)
    xgb_model.fit(X_train, y_train)

    # Iterative forecasting: use the last known row to forecast the next 'rolling_window' timesteps.
    last_known = current_train_df.iloc[-1]
    current_features = last_known[feature_cols].values.copy()
    preds = []
    for i in range(rolling_window):
        pred = xgb_model.predict(current_features.reshape(1, -1))[0]
        preds.append(pred)
        # Update features: shift and insert the new prediction at the beginning.
        current_features = np.roll(current_features, shift=1)
        current_features[0] = pred

    # If test_window has fewer than rolling_window timesteps, only use the corresponding number of predictions.
    preds = preds[:len(test_window)]
    
    # Record predictions and true values
    all_predictions.extend(preds)
    all_true_values.extend(test_window[target_col].values)
    prediction_timestamps.extend(test_window.index)

    # Update the training set with the entire test window (simulating live model updates)
    current_train_df = pd.concat([current_train_df, test_window])
# ---------------------------
# Evaluation Metrics
# ---------------------------
r2 = r2_score(all_true_values, all_predictions)
rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
print("XGBoost Iterative Forecasting Metrics:")
print(f"R2: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# Save metrics to CSV.
metrics_df = pd.DataFrame({
    "Metric": ["R2", "RMSE"],
    "Value": [r2, rmse]
})
metrics_csv_path = os.path.join(output_dir, "metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved metrics to {metrics_csv_path}")

# ---------------------------
# Visualization: Plots
# ---------------------------
# Convert predictions and true values to Series with timestamps.
pred_series = pd.Series(all_predictions, index=prediction_timestamps, name="Predicted")
true_series = pd.Series(all_true_values, index=prediction_timestamps, name="Actual")

# Plot 1: True vs. Predicted Prices
plt.figure(figsize=(12,6))
plt.plot(true_series.index, true_series, label="Actual", color="blue")
plt.plot(pred_series.index, pred_series, label="Predicted", color="red", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("XGBoost: Actual vs. Predicted Imbalance Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(output_dir, "true_vs_predicted.png")
plt.savefig(plot_path)
plt.show()
print(f"Saved plot to {plot_path}")

# Plot 2: Residual Histogram
errors = true_series - pred_series
plt.figure(figsize=(10,5))
plt.hist(errors, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Forecast Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Histogram of Forecast Errors (XGBoost)")
plt.grid(True)
plt.tight_layout()
plot_path2 = os.path.join(output_dir, "residual_histogram.png")
plt.savefig(plot_path2)
plt.show()
print(f"Saved plot to {plot_path2}")

# Plot 3: Absolute Error Over Time
absolute_errors = np.abs(errors)
plt.figure(figsize=(12,6))
plt.plot(true_series.index, absolute_errors, marker="o", linestyle="-", color="purple")
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.title("Absolute Forecast Error Over Time (XGBoost)")
plt.grid(True)
plt.tight_layout()
plot_path3 = os.path.join(output_dir, "absolute_error_over_time.png")
plt.savefig(plot_path3)
plt.show()
print(f"Saved plot to {plot_path3}")