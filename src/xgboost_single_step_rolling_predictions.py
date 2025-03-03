import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

# Import data extraction methods and curve definitions
from data.data_collection import get_data
from data.curves import curve_collections
import volue_insight_timeseries

# ---------------------------
# Setup Output Folder
# ---------------------------
output_dir = os.path.join("results", "xgboost_rolling_predictions_tuned_debug_v2")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Parameters and Data Loading
# ---------------------------
start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()

# Adjust the target curve as needed for your specific imbalance price
target_curve = curve_collections["de"]["mfrr"][0]  # e.g. "mfrr_up_price" or similar

session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

# If you only want the target, pass an empty list for X_curve_names
_, y_data, _, _ = get_data([], [target_curve], session, start_date, end_date)

# Convert y_data to a DataFrame with a 15-min frequency index (adjust if needed)
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
df.dropna(inplace=True)

feature_cols = [f"{target_curve}_lag{lag}" for lag in range(1, num_lags + 1)]
target_col = target_curve

# ---------------------------
# Rolling-Origin Setup
# ---------------------------
# Use first 80% of data for initial training, last 20% for rolling multi-step forecasts
split_fraction = 0.8
split_idx = int(len(df) * split_fraction)
df_train_initial = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

# Total number of blocks in test set
total_blocks = math.ceil(len(df_test) / 32)

all_predictions = []
all_true_values = []
prediction_timestamps = []

current_train_df = df_train_initial.copy()

# Forecast horizon: 32 timesteps (8 hours if each step is 15 min)
rolling_window = 32

# ---------------------------
# Iterative Multi-Step Forecasting: Rolling-Origin Validation
# ---------------------------
block_count = 0
for start in range(0, len(df_test), rolling_window):
    end = start + rolling_window
    test_window = df_test.iloc[start:end].copy()
    if test_window.empty:
        break

    block_count += 1

    # ---------------------------
    # 1) Prepare Training Data
    # ---------------------------
    X_train = current_train_df[feature_cols].values
    y_train = current_train_df[target_col].values

    # Create an eval_set for training progress
    eval_set = [(X_train, y_train)]

    # ---------------------------
    # 2) Train XGBoost Model with Tuned Parameters
    # ---------------------------
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=20,
        random_state=42
    )

    print(f"\n[Block {block_count}/{total_blocks}] Training XGBoost on {len(X_train)} samples...")
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=100
    )

    # ---------------------------
    # 3) Iterative Forecasting for This Block
    # ---------------------------
    # Start from the last known row in current_train_df
    last_known = current_train_df.iloc[-1]
    current_features = last_known[feature_cols].values.copy()

    block_preds = []
    for i in range(rolling_window):
        pred = xgb_model.predict(current_features.reshape(1, -1))[0]
        block_preds.append(pred)
        # Update features: shift and insert new prediction at index 0
        current_features = np.roll(current_features, shift=1)
        current_features[0] = pred

    # If test_window is shorter than rolling_window, adjust predictions
    block_preds = block_preds[: len(test_window)]
    block_true = test_window[target_col].values[: len(block_preds)]
    block_index = test_window.index[: len(block_preds)]

    # ---------------------------
    # 4) Evaluate and Print Block Metrics with Debug Info
    # ---------------------------
    block_rmse = np.sqrt(mean_squared_error(block_true, block_preds)) if len(block_preds) > 1 else 0
    block_mae = mean_absolute_error(block_true, block_preds) if len(block_preds) > 1 else 0

    # Only compute R² if the variance of true values is sufficiently high
    if np.std(block_true) > 1e-3:
        block_r2 = r2_score(block_true, block_preds)
    else:
        block_r2 = None

    if block_r2 is None:
        print(f"[Block {block_count}/{total_blocks}] True values variance nearly zero. R² not reliable.")
    else:
        print(f"[Block {block_count}/{total_blocks}] R² on this block: {block_r2:.3f}")
    print(f"[Block {block_count}/{total_blocks}] RMSE: {block_rmse:.3f}, MAE: {block_mae:.3f}")
    print(f"[Block {block_count}/{total_blocks}] True values: mean={np.mean(block_true):.3f}, std={np.std(block_true):.3f}")
    print(f"[Block {block_count}/{total_blocks}] Predicted values: mean={np.mean(block_preds):.3f}, std={np.std(block_preds):.3f}")

    # Store block results for global evaluation
    all_predictions.extend(block_preds)
    all_true_values.extend(block_true)
    prediction_timestamps.extend(block_index)

    # ---------------------------
    # 5) Update Training Set
    # ---------------------------
    current_train_df = pd.concat([current_train_df, test_window])

# ---------------------------
# Final Overall Metrics
# ---------------------------
final_rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
final_mae = mean_absolute_error(all_true_values, all_predictions)
if np.std(all_true_values) > 1e-3:
    final_r2 = r2_score(all_true_values, all_predictions)
else:
    final_r2 = None

print("\n=======================")
print("Final Overall Metrics:")
if final_r2 is None:
    print("R²: Not reliable (true values variance nearly zero)")
else:
    print(f"R²: {final_r2:.3f}")
print(f"RMSE: {final_rmse:.3f}")
print(f"MAE: {final_mae:.3f}")

# ---------------------------
# Save Metrics and Visualize
# ---------------------------
metrics_data = {
    "RMSE": final_rmse,
    "MAE": final_mae
}
if final_r2 is not None:
    metrics_data["R2"] = final_r2
else:
    metrics_data["R2"] = "Not reliable (low variance)"
    
metrics_df = pd.DataFrame(list(metrics_data.items()), columns=["Metric", "Value"])
metrics_csv_path = os.path.join(output_dir, "metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved final metrics to {metrics_csv_path}")

pred_series = pd.Series(all_predictions, index=prediction_timestamps, name="Predicted")
true_series = pd.Series(all_true_values, index=prediction_timestamps, name="Actual")

# Plot 1: True vs. Predicted
plt.figure(figsize=(12,6))
plt.plot(true_series.index, true_series, label="Actual", color="blue")
plt.plot(pred_series.index, pred_series, label="Predicted", color="red", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("XGBoost (Tuned) Rolling Multi-Step: Actual vs. Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(output_dir, "true_vs_predicted_rolling_tuned.png")
plt.savefig(plot_path)
plt.show()

# Plot 2: Residual Histogram
errors = true_series - pred_series
plt.figure(figsize=(10,5))
plt.hist(errors, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Forecast Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Histogram of Forecast Errors (Tuned Rolling Multi-Step)")
plt.grid(True)
plt.tight_layout()
plot_path2 = os.path.join(output_dir, "residual_histogram_rolling_tuned.png")
plt.savefig(plot_path2)
plt.show()

# Plot 3: Absolute Error Over Time
absolute_errors = np.abs(errors)
plt.figure(figsize=(12,6))
plt.plot(true_series.index, absolute_errors, marker="o", linestyle="-", color="purple")
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.title("Absolute Forecast Error Over Time (Tuned Rolling Multi-Step)")
plt.grid(True)
plt.tight_layout()
plot_path3 = os.path.join(output_dir, "absolute_error_over_time_rolling_tuned.png")
plt.savefig(plot_path3)
plt.show()

print(f"Saved plots to {output_dir}")
