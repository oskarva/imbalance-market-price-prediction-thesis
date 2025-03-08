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

X, y, X_columns, y_columns = get_data(X_curve_names, [target_curve],
                                      session,
                                      start_date, end_date,
                                      add_time=False, 
                                      add_lag=True,
                                      add_rolling=False,
                                      )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ---------------------------
# Model training loop
# ---------------------------
# ----------------------
# Revised approach
# ----------------------
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Check for data leakage or temporal issues
print("Checking temporal patterns...")
plt.figure(figsize=(12, 6))
plt.plot(y_train, label='Training Data')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Data')
plt.title('Target Value Over Time')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.savefig('temporal_pattern.png')
plt.close()

# 2. Train a different model with more appropriate parameters
print("\nTraining a gradient boosting model...")
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train.ravel())  # Use ravel() to convert to 1D array

# 3. Evaluate on test set
print("Evaluating single-step predictions...")
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Single-step metrics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# 4. Fixed iterative forecast function
def iterative_forecast(model, X_initial, steps=32, target_lag_idx=None):
    """
    Iterative forecasting without scaling complications
    """
    predictions = []
    current_features = X_initial.copy().reshape(1, -1)  # Ensure 2D
    
    for step in range(steps):
        # Predict next value
        next_pred = model.predict(current_features)[0]
        predictions.append(next_pred)
        
        if target_lag_idx is not None:
            # Create a copy of the current features for the next step
            next_features = current_features.copy()
            # Update the target lag column with our prediction
            next_features[0, target_lag_idx] = next_pred
            # Set up for next iteration
            current_features = next_features
    
    return np.array(predictions)

# Find target lag index 
target_col = y_columns[0]
target_lag_idx = None
for i, col in enumerate(X_columns):
    if target_col in col and "_lag_" in col:
        print(f"Found target lag column: {col} at index {i}")
        target_lag_idx = i
        break

# Try multi-step forecasting on a few examples
if target_lag_idx is not None:
    print("\nTesting iterative forecasting...")
    # Select 3 test points evenly spaced
    test_indices = [0, len(X_test)//3, 2*len(X_test)//3]
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(test_indices):
        initial_features = X_test[idx]
        actual_sequence = y_test[idx:idx+32].ravel()
        
        # Skip if we don't have enough actual values
        if len(actual_sequence) < 32:
            continue
            
        predicted_sequence = iterative_forecast(
            model, 
            initial_features, 
            steps=min(32, len(actual_sequence)),
            target_lag_idx=target_lag_idx
        )
        
        # Plot this sequence
        plt.subplot(3, 1, i+1)
        plt.plot(actual_sequence, 'b-', label='Actual')
        plt.plot(predicted_sequence, 'r--', label='Predicted')
        plt.title(f'Multi-Step Forecast (Starting at index {idx})')
        plt.xlabel('Steps (15-min intervals)')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('multi_step_forecasts.png')
    plt.show()

# Feature importance analysis
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [X_columns[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.show()