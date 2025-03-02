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
output_dir = os.path.join("results", "xgboost_single_step_extended")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Parameters and Data Loading
# ---------------------------
start_date = pd.Timestamp("2021-01-01")  # Now starting from Jan 1, 2021
end_date = pd.Timestamp.today()

# Adjust the target curve as needed for the specific imbalance price
target_curve = curve_collections["de"]["mfrr"][0]  

session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))
# If only want the target, pass empty list for X_curve_names
_, y_data, _, _ = get_data([], [target_curve], session, start_date, end_date)

# Convert y_data to a DataFrame with a 15-min frequency index 
if isinstance(y_data, np.ndarray):
    time_index = pd.date_range(start=start_date, periods=len(y_data), freq='15min')
    df = pd.DataFrame(y_data, index=time_index, columns=[target_curve])
else:
    df = y_data.copy().sort_index()

# ---------------------------
# Feature Engineering: Create Lag Features
# ---------------------------
def create_lag_features(df, target, num_lags):
    """
    Create lag features for the target column.
    For each lag in 1..num_lags, a new column 'target_lag{lag}' is created.
    """
    for lag in range(1, num_lags + 1):
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df

num_lags = 8
df = create_lag_features(df, target_curve, num_lags)

# Drop rows with NaNs introduced by shifting
df.dropna(inplace=True)

feature_cols = [f"{target_curve}_lag{lag}" for lag in range(1, num_lags + 1)]
target_col = target_curve

# ---------------------------
# Rolling-Origin Setup (Single-Step)
# ---------------------------
# Use first 80% of data as the initial training set (adjust if you prefer 90/10, etc.)
split_fraction = 0.8
split_idx = int(len(df) * split_fraction)
df_train_initial = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

all_predictions = []
all_true_values = []
prediction_timestamps = []

current_train_df = df_train_initial.copy()

# ---------------------------
# Single-Step Rolling Forecast
# ---------------------------
for i in range(len(df_test)):
    # The 'test_row' is the next time step we want to predict
    test_row = df_test.iloc[i : i + 1]
    if test_row.empty:
        break

    # Prepare training data for this fold
    X_train = current_train_df[feature_cols].values
    y_train = current_train_df[target_col].values

    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(
        random_state=42,
        n_estimators=100,
        # You can tune other hyperparams if needed
    )
    xgb_model.fit(X_train, y_train)

    # Construct features for the test row
    X_test = test_row[feature_cols].values  # shape (1, num_lags)
    # Predict the next 15-min price
    pred = xgb_model.predict(X_test)[0]

    # Record the prediction
    all_predictions.append(pred)
    all_true_values.append(test_row[target_col].values[0])
    prediction_timestamps.append(test_row.index[0])

    # Optional debug: print the first few predictions
    if i < 5:
        print(
            f"Test timestamp: {test_row.index[0]}, "
            f"Prediction: {pred}, "
            f"Actual: {test_row[target_col].values[0]}"
        )

    # "Roll" forward: add the new actual row to the training set
    current_train_df = pd.concat([current_train_df, test_row])

# ---------------------------
# Evaluation Metrics
# ---------------------------
r2 = r2_score(all_true_values, all_predictions)
rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
print("\nXGBoost Single-Step Forecasting (Extended Training) Metrics:")
print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    "Metric": ["R2", "RMSE"],
    "Value": [r2, rmse]
})
metrics_csv_path = os.path.join(output_dir, "metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved metrics to {metrics_csv_path}")

# ---------------------------
# Visualization
# ---------------------------
pred_series = pd.Series(all_predictions, index=prediction_timestamps, name="Predicted")
true_series = pd.Series(all_true_values, index=prediction_timestamps, name="Actual")

# Plot 1: True vs. Predicted
plt.figure(figsize=(12,6))
plt.plot(true_series.index, true_series, label="Actual", color="blue")
plt.plot(pred_series.index, pred_series, label="Predicted", color="red", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("XGBoost Single-Step: Actual vs. Predicted (Extended Training)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(output_dir, "true_vs_predicted_single_step_extended.png")
plt.savefig(plot_path)
plt.show()

# Plot 2: Residual Histogram
errors = true_series - pred_series
plt.figure(figsize=(10,5))
plt.hist(errors, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Forecast Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Histogram of Forecast Errors (Single-Step, Extended Training)")
plt.grid(True)
plt.tight_layout()
plot_path2 = os.path.join(output_dir, "residual_histogram_single_step_extended.png")
plt.savefig(plot_path2)
plt.show()

# Plot 3: Absolute Error Over Time
absolute_errors = np.abs(errors)
plt.figure(figsize=(12,6))
plt.plot(true_series.index, absolute_errors, marker="o", linestyle="-", color="purple")
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.title("Absolute Forecast Error Over Time (Single-Step, Extended Training)")
plt.grid(True)
plt.tight_layout()
plot_path3 = os.path.join(output_dir, "absolute_error_over_time_single_step_extended.png")
plt.savefig(plot_path3)
plt.show()

print(f"Saved plots to {output_dir}")
