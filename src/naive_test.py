import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import volue_insight_timeseries 
from data.curves import curve_collections
from data.data_collection import get_data
import matplotlib.pyplot as plt

# Import (or define) your naive model.
from models.naive import Naive_Last_Known_Activation_Price

# ---------------------------
# Create Output Folders
# ---------------------------
output_dir = os.path.join("results", "naive")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
    time_index = pd.date_range(start=start_date, periods=len(y), freq='H')
    y = pd.Series(y.flatten(), index=time_index, name=curve_collections["de"]["mfrr"][0])
else:
    y = y.sort_index()

# Instead of a random train/test split, use the first 10% of the data as initial training.
split_idx = int(len(y) * 0.1)
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# ---------------------------
# Rolling Forecast Settings
# ---------------------------
rolling_window = 8  # Forecast 8 hours ahead in each rolling iteration.

predictions = []
true_values = []
current_train_y = y_train.copy()

for start in range(0, len(y_test), rolling_window):
    end = start + rolling_window
    test_window = y_test.iloc[start:end]
    if test_window.empty:
        break

    model = Naive_Last_Known_Activation_Price()
    model.train(current_train_y.to_frame(), column_name=curve_collections["de"]["mfrr"][0])
    
    pred = model.predict()
    pred_array = np.full(len(test_window), pred)
    
    predictions.extend(pred_array)
    true_values.extend(test_window.values)
    
    current_train_y = pd.concat([current_train_y, test_window])

# ---------------------------
# Final Evaluation Metrics
# ---------------------------
final_r2 = r2_score(true_values, predictions)
final_rmse = np.sqrt(mean_squared_error(true_values, predictions))

print("Final R2:", final_r2)
print("Final RMSE:", final_rmse)

# Save metrics to CSV.
metrics_df = pd.DataFrame({
    "Metric": ["R2", "RMSE"],
    "Value": [final_r2, final_rmse]
})
metrics_csv_path = os.path.join(output_dir, "metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved metrics to {metrics_csv_path}")

# Create predictions Series using the same index as y_test.
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
plot1_path = os.path.join(output_dir, "true_vs_predicted.png")
plt.savefig(plot1_path)
print(f"Saved plot to {plot1_path}")
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
plot2_path = os.path.join(output_dir, "residual_histogram.png")
plt.savefig(plot2_path)
print(f"Saved plot to {plot2_path}")
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
plot3_path = os.path.join(output_dir, "absolute_error_over_time.png")
plt.savefig(plot3_path)
print(f"Saved plot to {plot3_path}")
plt.show()