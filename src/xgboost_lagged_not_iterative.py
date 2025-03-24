import os
import math
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

# Import data extraction methods and curve definitions
from data.data_collection import get_data
from data.curves import curve_collections
import volue_insight_timeseries

# ---------------------------
# Setup Output Folder
# ---------------------------
output_dir = os.path.join("results", "xgboost_one_shot_rolling")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Parameters and Data Loading
# ---------------------------
start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()

# Define curves:
# X_curve_names: additional independent curves.
# target_curve: the target variable.
X_curve_names = curve_collections["de"]["X"]    # additional curves
target_curve = curve_collections["de"]["mfrr"][0]  # e.g. "mfrr_up_price" or similar

session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

# Get both X curves and the target curve.
X_data, y_data, X_col, _ = get_data(X_curve_names, [target_curve], session, start_date, end_date)

# Convert X_data to DataFrame if needed.
if isinstance(X_data, np.ndarray):
    time_index = pd.date_range(start=start_date, periods=len(X_data), freq='15min')
    df_X = pd.DataFrame(X_data, index=time_index, columns=X_col)
else:
    df_X = X_data.copy().sort_index()

# Convert y_data to DataFrame if needed.
if isinstance(y_data, np.ndarray):
    time_index = pd.date_range(start=start_date, periods=len(y_data), freq='15min')
    df_y = pd.DataFrame(y_data, index=time_index, columns=[target_curve])
else:
    df_y = y_data.copy().sort_index()

# Merge X and y DataFrames.
if target_curve in df_X.columns:
    df_X.rename(columns={target_curve: target_curve + "_x"}, inplace=True)
df = pd.concat([df_X, df_y], axis=1)

# ---------------------------
# Feature Engineering: Create Custom-Shifted Features
# ---------------------------
def create_custom_shifted_features(df, shift_value):
    """
    For each column in df, create a shifted column using shift_value.
    Returns a new DataFrame containing ONLY these shifted features.
    """
    df_shifted = pd.DataFrame(index=df.index)
    for col in df.columns:
        df_shifted[f"{col}_shift{shift_value}"] = df[col].shift(shift_value)
    return df_shifted

# Use a shift of 32 for all variables (including the target)
shift_val = 32
df_features = create_custom_shifted_features(df, shift_val)
df_features.dropna(inplace=True)

# ---------------------------
# Create Multi-Step Labels
# ---------------------------
# We want a one-shot forecast of the next 32 timesteps for the target.
horizon = 32
def create_multi_step_labels(series, indices, horizon, freq_minutes=15):
    """
    For each timestamp in 'indices', extract the next 'horizon' values
    from the target series. Returns a 2D NumPy array (n_samples x horizon)
    and a list of valid indices.
    """
    labels = []
    valid_idx = []
    for t in indices:
        end_time = t + pd.Timedelta(minutes=freq_minutes*(horizon)-freq_minutes)
        window = series.loc[t:end_time]
        if len(window) == horizon:
            labels.append(window.values.flatten())
            valid_idx.append(t)
    return np.array(labels), valid_idx

# Create multi-step labels from the unshifted target (df_y)
Y_all, valid_indices = create_multi_step_labels(df_y[target_curve], df_features.index, horizon, freq_minutes=15)
# Keep only the rows with valid labels.
df_features = df_features.loc[valid_indices]
print(f"Total samples after multi-step label creation: {len(df_features)}")

# ---------------------------
# Split Data: Use 95% for training, 5% for test (rolling)
# ---------------------------
split_fraction = 0.95
split_idx = int(len(df_features) * split_fraction)
X_all = df_features.values   # features: all columns shifted by 32
Y_all = Y_all                # multi-step labels: each row is a vector of length horizon

X_train_initial = X_all[:split_idx]
Y_train_initial = Y_all[:split_idx]
X_test = X_all[split_idx:]
Y_test = Y_all[split_idx:]
test_indices = df_features.index[split_idx:]
print(f"Training samples: {len(X_train_initial)}, Testing samples (blocks will be processed): {len(X_test)}")

# ---------------------------
# Rolling-One-Shot Multi-Step Forecasting
# ---------------------------
# We will process the test set in blocks of 'horizon' samples.
total_blocks = math.ceil(len(X_test) / horizon)
all_predictions = []
all_true_values = []
prediction_timestamps = []

# Initialize current training set with initial training samples.
current_train_features = X_train_initial.copy()
current_train_labels = Y_train_initial.copy()

block_count = 0
for start in range(0, len(X_test), horizon):
    end = start + horizon
    # Get the test block for this iteration.
    block_X = X_test[start:end]
    block_Y = Y_test[start:end]
    block_indices = test_indices[start:end]
    if len(block_X) == 0:
        break
    block_count += 1
    print(f"\n[Block {block_count}/{total_blocks}] Training multi-output model on {len(current_train_features)} samples...")

    # Train multi-output model on the current training set.
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=20,
        random_state=42
    )
    model = MultiOutputRegressor(base_model)
    model.fit(current_train_features, current_train_labels)

    # Predict one-shot multi-step forecast for this block.
    block_preds = model.predict(block_X)  # shape: (n_block_samples, horizon)
    # For evaluation, flatten predictions and true values.
    all_predictions.extend(block_preds.flatten())
    all_true_values.extend(block_Y.flatten())
    prediction_timestamps.extend(block_indices.repeat(horizon)[:block_preds.flatten().shape[0]])

    # Print block metrics (using flattened values)
    block_r2 = r2_score(block_Y.flatten(), block_preds.flatten())
    block_rmse = np.sqrt(mean_squared_error(block_Y.flatten(), block_preds.flatten()))
    block_mae = mean_absolute_error(block_Y.flatten(), block_preds.flatten())
    print(f"[Block {block_count}/{total_blocks}] R²: {block_r2:.3f}, RMSE: {block_rmse:.3f}, MAE: {block_mae:.3f}")

    # Update the training set by appending the test block.
    # In a real-world scenario, would only append the new actual data.
    # Here we append both features and multi-step labels for this block.
    current_train_features = np.concatenate([current_train_features, block_X], axis=0)
    current_train_labels = np.concatenate([current_train_labels, block_Y], axis=0)

# ---------------------------
# Final Overall Metrics
# ---------------------------
all_predictions = np.array(all_predictions)
all_true_values = np.array(all_true_values)
final_r2 = r2_score(all_true_values, all_predictions)
final_rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
final_mae = mean_absolute_error(all_true_values, all_predictions)
print("\n=======================")
print("Final Overall Metrics (One-Shot Multi-Step Rolling):")
print(f"R²: {final_r2:.3f}")
print(f"RMSE: {final_rmse:.3f}")
print(f"MAE: {final_mae:.3f}")

# ---------------------------
# Save Metrics
# ---------------------------
metrics_data = {
    "R2": final_r2,
    "RMSE": final_rmse,
    "MAE": final_mae
}
import json
metrics_csv_path = os.path.join(output_dir, "metrics.json")
with open(metrics_csv_path, "w") as f:
    json.dump(metrics_data, f, indent=2)
print(f"Saved final metrics to {metrics_csv_path}")

# ---------------------------
# Visualization: Plot One Sample's Multi-Step Forecast from the Final Block
# ---------------------------
# Plot the first sample of the last block.
if len(block_X) > 0:
    sample_idx = 0
    sample_true = block_Y[sample_idx]
    sample_pred = block_preds[sample_idx]
    time_axis = np.arange(1, horizon+1)
    plt.figure(figsize=(10,5))
    plt.plot(time_axis, sample_true, marker='o', label="Actual")
    plt.plot(time_axis, sample_pred, marker='x', label="Predicted", linestyle="--")
    plt.xlabel("Forecast Timestep (15-min intervals)")
    plt.ylabel("Target Value")
    plt.title("One-Shot Multi-Step Forecast for One Test Sample (Final Block)")
    plt.legend()
    plt.grid(True)
    sample_plot_path = os.path.join(output_dir, "sample_forecast_final_block.png")
    plt.savefig(sample_plot_path)
    plt.show()
    print(f"Saved sample forecast plot to {sample_plot_path}")
