import os
import math
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import data extraction methods and curve definitions
from data.data_collection import get_data
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

# Get both X curves and the target curve.
X_data, y_data, X_col, _ = get_data(X_curve_names, [target_curve], session, start_date, end_date)

# Convert X_data to a DataFrame if needed.
if isinstance(X_data, np.ndarray):
    time_index = pd.date_range(start=start_date, periods=len(X_data), freq='15min')
    df_X = pd.DataFrame(X_data, index=time_index, columns=X_col)
else:
    df_X = X_data.copy().sort_index()

# Convert y_data to a DataFrame if needed.
if isinstance(y_data, np.ndarray):
    time_index = pd.date_range(start=start_date, periods=len(y_data), freq='15min')
    df_y = pd.DataFrame(y_data, index=time_index, columns=[target_curve])
else:
    df_y = y_data.copy().sort_index()

# Merge the X curves and target curve into one DataFrame.
# We assume they share the same time index.
# If the target appears in df_X, rename it to avoid collision.
if target_curve in df_X.columns:
    df_X.rename(columns={target_curve: target_curve + "_x"}, inplace=True)
df = pd.concat([df_X, df_y], axis=1)

# ---------------------------
# Feature Engineering: Create Custom-Shifted Features
# ---------------------------
def create_custom_shifted_features(df, target, shift_target=1, shift_others=32):
    """
    For each column in df, create a shifted version:
      - For the target variable, shift by shift_target.
      - For all other variables, shift by shift_others.
    Returns a new DataFrame containing ONLY the shifted features.
    """
    df_shifted = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col == target:
            df_shifted[f"{col}_shift{shift_target}"] = df[col].shift(shift_target)
        else:
            df_shifted[f"{col}_shift{shift_others}"] = df[col].shift(shift_others)
    return df_shifted

# Create the shifted features:
# - The target variable gets a 1-step shift.
# - All other variables get a 32-step shift.
df_features = create_custom_shifted_features(df, target=target_curve, shift_target=1, shift_others=32)
df_features.dropna(inplace=True)

# The training label remains the original target variable from df_y.
df_label = df_y.loc[df_features.index, target_curve]

# ---------------------------
# Rolling-Origin Setup
# ---------------------------
# Use first 80% of the available (shifted) data for initial training; the remaining for rolling forecasts.
split_fraction = 0.8
split_idx = int(len(df_features) * split_fraction)
df_train_initial = df_features.iloc[:split_idx].copy()
df_test = df_features.iloc[split_idx:].copy()

# Similarly, split the labels.
y_train_initial = df_label.iloc[:split_idx].copy()
y_test = df_label.iloc[split_idx:].copy()

# Total number of blocks (forecast horizon: 32 timesteps per block)
rolling_window = 32  
total_blocks = math.ceil(len(df_test) / rolling_window)

all_predictions = []
all_true_values = []
prediction_timestamps = []

# Start with initial training DataFrame (features and label).
current_train_features = df_train_initial.copy()
current_train_labels = y_train_initial.copy()

# ---------------------------
# Iterative Multi-Step Forecasting: Rolling-Origin Validation
# ---------------------------
block_count = 0
for start in range(0, len(df_test), rolling_window):
    end = start + rolling_window
    test_window = df_test.iloc[start:end].copy()  # shifted features for test block
    test_labels = y_test.iloc[start:end].copy()     # true labels for test block
    if test_window.empty:
        break

    block_count += 1

    # Prepare training data for this block
    X_train = current_train_features.values
    y_train = current_train_labels.values

    # Create an eval_set for training progress
    eval_set = [(X_train, y_train)]

    # Train XGBoost model with tuned parameters
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
    # Iterative Forecasting for This Block
    # ---------------------------
    # For iterative forecasting:
    # - The feature vector for each independent variable is the 32-shifted column.
    # - The target variable feature (1-shifted) will be updated with new forecast value.
    # Start from the last row of the current training features.
    last_known = current_train_features.iloc[-1]
    current_features = last_known.values.copy()  # 1D array of features

    block_preds = []
    for i in range(rolling_window):
        pred = xgb_model.predict(current_features.reshape(1, -1))[0]
        block_preds.append(pred)
        # Identify index of target variable's shifted feature.
        feature_names = current_train_features.columns.tolist()
        target_feature_name = f"{target_curve}_shift1"
        target_index = feature_names.index(target_feature_name)
        # Update: shift all values, then insert new forecast at target's index.
        current_features = np.roll(current_features, shift=1)
        current_features[target_index] = pred

    # If test_window has fewer than rolling_window rows, adjust predictions.
    block_preds = block_preds[:len(test_window)]
    block_true = test_labels.values[:len(block_preds)]
    block_index = test_window.index[:len(block_preds)]

    # ---------------------------
    # Evaluate and Print Block Metrics with Debug Info
    # ---------------------------
    block_rmse = np.sqrt(mean_squared_error(block_true, block_preds)) if len(block_preds) > 1 else 0
    block_mae = mean_absolute_error(block_true, block_preds) if len(block_preds) > 1 else 0
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
    # Update Training Set: Append the test block (features and corresponding label)
    # ---------------------------
    current_train_features = pd.concat([current_train_features, test_window])
    current_train_labels = pd.concat([current_train_labels, df_y.loc[test_window.index, target_curve]])

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
