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
                                      curve_collections["de"]["X_to_forecast"],
                                      add_time=False, 
                                      add_lag=False,
                                      add_rolling=False,
                                      )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ---------------------------
# Model training loop
# ---------------------------
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

# Define target_lag_idx early and explicitly
target_lag_idx = None
target_col = y_columns[0]
print(f"Looking for lag column for target: {target_col}")

for i, col in enumerate(X_columns):
    if target_col in col and "_lag_" in col:
        target_lag_idx = i
        print(f"Found target lag column: {col} at index {i}")
        break

if target_lag_idx is None:
    print("WARNING: Could not find target lag column in features. Iterative forecasting may not work correctly.")

# 2. Train a different model with more appropriate parameters
print("\nTraining a gradient boosting model...")
from sklearn.ensemble import GradientBoostingRegressor
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,      # number of boosting rounds
    learning_rate=0.01,     # step size shrinkage used in update to prevent overfitting
    max_depth=20,           # maximum depth of a tree
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

# Try multi-step forecasting with metrics calculation
if target_lag_idx is not None:  # Check that we found a target lag column
    print("\nEvaluating iterative forecasting performance...")
    test_indices = [0, len(X_test)//3, 2*len(X_test)//3]
    
    all_actuals = []
    all_predictions = []
    
    plt.figure(figsize=(15, 15))
    
    for i, idx in enumerate(test_indices):
        initial_features = X_test[idx]
        horizon = 32
        
        # Ensure we have enough data
        if idx + horizon > len(y_test):
            horizon = len(y_test) - idx
            
        actual_sequence = y_test[idx:idx+horizon].ravel()
        
        print(f"Forecasting sequence {i+1} starting at index {idx} for {horizon} steps")
        print(f"Target lag index being used: {target_lag_idx}")
        
        # Generate forecasts
        predicted_sequence = iterative_forecast(
            model, 
            initial_features, 
            steps=horizon,
            target_lag_idx=target_lag_idx
        )
        
        # Store for metrics calculation
        all_actuals.append(actual_sequence)
        all_predictions.append(predicted_sequence)
        
        # Calculate metrics for this sequence
        mae_seq = mean_absolute_error(actual_sequence, predicted_sequence)
        rmse_seq = np.sqrt(mean_squared_error(actual_sequence, predicted_sequence))
        r2_seq = r2_score(actual_sequence, predicted_sequence)
        
        print(f"\nForecast sequence {i+1} (starting at index {idx}):")
        print(f"MAE: {mae_seq:.2f}")
        print(f"RMSE: {rmse_seq:.2f}")
        print(f"R²: {r2_seq:.4f}")
        
        # Plot this sequence
        plt.subplot(3, 1, i+1)
        plt.plot(actual_sequence, 'b-', label='Actual')
        plt.plot(predicted_sequence, 'r--', label='Predicted')
        plt.title(f'Multi-Step Forecast {i+1} (MAE: {mae_seq:.2f}, RMSE: {rmse_seq:.2f})')
        plt.xlabel('Steps (15-min intervals)')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
    
    # Calculate overall metrics
    all_actuals_flat = np.concatenate(all_actuals)
    all_predictions_flat = np.concatenate(all_predictions)
    
    mae_overall = mean_absolute_error(all_actuals_flat, all_predictions_flat)
    rmse_overall = np.sqrt(mean_squared_error(all_actuals_flat, all_predictions_flat))
    r2_overall = r2_score(all_actuals_flat, all_predictions_flat)
    
    print("\nOverall iterative forecasting metrics:")
    print(f"MAE: {mae_overall:.2f}")
    print(f"RMSE: {rmse_overall:.2f}")
    print(f"R²: {r2_overall:.4f}")
    
    plt.tight_layout()
    plt.savefig('multi_step_forecasts.png')
    plt.show()
else:
    print("Skipping iterative forecasting because target lag index was not found")

# Add a simple baseline comparison (predict previous value)
print("\n--- Comparing with baseline (naive forecast) ---")
y_naive = np.roll(y_test, 1)  # Shift by 1 to predict previous value
y_naive[0] = y_naive[1]  # Handle the first value

mae_naive = mean_absolute_error(y_test, y_naive)
rmse_naive = np.sqrt(mean_squared_error(y_test, y_naive))
r2_naive = r2_score(y_test, y_naive)

print(f"Naive forecast (previous value) metrics:")
print(f"MAE: {mae_naive:.2f}")
print(f"RMSE: {rmse_naive:.2f}")
print(f"R²: {r2_naive:.4f}")

# Print correlation between features and target
print("\n--- Feature correlations with target ---")
X_df = pd.DataFrame(X_train, columns=X_columns)
X_df['target'] = y_train
correlations = X_df.corr()['target'].sort_values(ascending=False)
print(correlations)

# Code to plot all iterative predictions vs. actual values in a clear time series

# First, let's run multiple forecasts across the test set
def create_comprehensive_forecast_visualization(X_test, y_test, model, target_lag_idx, forecast_horizon=32):
    # Define number of forecasts to make
    num_forecasts = 10
    forecast_spacing = max(1, len(X_test) // num_forecasts)
    
    # Store results
    all_indices = []
    all_actuals = []
    all_predictions = []
    
    plt.figure(figsize=(15, 10))
    
    # Top plot: Overall view
    plt.subplot(211)
    plt.plot(y_test, 'b-', alpha=0.5, label='Actual Values')
    
    # Generate multiple forecasts across the test set
    for i in range(num_forecasts):
        start_idx = i * forecast_spacing
        if start_idx + forecast_horizon >= len(X_test):
            break
            
        # Get features and actual values
        initial_features = X_test[start_idx]
        actual_sequence = y_test[start_idx:start_idx+forecast_horizon].ravel()
        
        # Generate forecast
        predicted_sequence = iterative_forecast(
            model, 
            initial_features, 
            steps=forecast_horizon,
            target_lag_idx=target_lag_idx
        )
        
        # Store for later use
        all_indices.append(np.arange(start_idx, start_idx+forecast_horizon))
        all_actuals.append(actual_sequence)
        all_predictions.append(predicted_sequence)
        
        # Plot this forecast sequence
        forecast_indices = np.arange(start_idx, start_idx+forecast_horizon)
        plt.plot(forecast_indices, predicted_sequence, 'r-', alpha=0.7)
    
    plt.title('Multiple Iterative Forecasts (Red) vs. Actual Values (Blue)')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Bottom plot: Detail view of a specific forecast
    plt.subplot(212)
    selected_forecast = len(all_predictions) // 2  # Pick a middle forecast
    
    if selected_forecast < len(all_predictions):
        selected_idx = all_indices[selected_forecast][0]
        window_size = forecast_horizon * 2
        
        # Show actual values around the forecast
        display_start = max(0, selected_idx - forecast_horizon//2)
        display_end = min(len(y_test), selected_idx + forecast_horizon + forecast_horizon//2)
        
        plt.plot(np.arange(display_start, display_end), 
                 y_test[display_start:display_end], 'b-', label='Actual')
        
        # Show the forecast
        plt.plot(all_indices[selected_forecast], 
                 all_predictions[selected_forecast], 'r-', label='Predicted')
        
        plt.title(f'Detailed View of Forecast {selected_forecast+1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('comprehensive_forecast_visualization.png')
    plt.show()
    
    # Calculate and print overall metrics
    all_actuals_flat = np.concatenate(all_actuals)
    all_predictions_flat = np.concatenate(all_predictions)
    
    mae_overall = mean_absolute_error(all_actuals_flat, all_predictions_flat)
    rmse_overall = np.sqrt(mean_squared_error(all_actuals_flat, all_predictions_flat))
    r2_overall = r2_score(all_actuals_flat, all_predictions_flat)
    
    print("\nComprehensive forecasting metrics (across multiple sequences):")
    print(f"MAE: {mae_overall:.2f}")
    print(f"RMSE: {rmse_overall:.2f}")
    print(f"R²: {r2_overall:.4f}")
    
    return all_indices, all_actuals, all_predictions

# Run the comprehensive visualization
indices, actuals, predictions = create_comprehensive_forecast_visualization(
    X_test, y_test, model, target_lag_idx)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# Function to create lag features (in case we need to recreate them during retraining)
def prepare_features(X, y, lag=32):
    # Keep track of any feature creation logic here
    # This function might be needed if you need to recreate features during retraining
    return X, y

# Implement the rolling origin forecast evaluation
def rolling_origin_forecast(X, y, forecast_horizon=32, target_lag_idx=13):
    """
    Implements rolling origin forecasting with model retraining.
    
    Args:
        X: Feature matrix
        y: Target vector
        forecast_horizon: Number of steps to forecast
        target_lag_idx: Index of the target lag feature
    """
    # Create initial train/test split (95/5)
    train_size = int(len(X) * 0.99)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Storage for results
    all_forecasts = []
    all_actuals = []
    forecast_indices = []
    training_times = []
    
    # Initial model training
    print(f"Initial training on {len(X_train)} samples...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,      # number of boosting rounds
        learning_rate=0.01,     # step size shrinkage used in update to prevent overfitting
        max_depth=20,           # maximum depth of a tree
        random_state=42
    )
    start_time = time.time()
    model.fit(X_train, y_train.ravel())
    training_times.append(time.time() - start_time)
    
    # Number of forecast iterations
    num_iterations = (len(X_test) - forecast_horizon) // forecast_horizon + 1
    print(f"Will perform {num_iterations} forecast iterations...")
    
    for i in range(num_iterations):
        current_idx = train_size + i * forecast_horizon
        print(f"\nForecast iteration {i+1}/{num_iterations} (starting at idx {current_idx})")
        
        # Extract current features for forecasting
        if current_idx < len(X):
            current_features = X[current_idx]
            
            # Get actual values for this period
            end_idx = min(current_idx + forecast_horizon, len(y))
            actual_values = y[current_idx:end_idx].ravel()
            
            # Generate iterative forecast
            print(f"  Forecasting {len(actual_values)} steps ahead...")
            predicted_values = iterative_forecast(
                model, 
                current_features,
                steps=len(actual_values),
                target_lag_idx=target_lag_idx
            )
            
            # Store results
            all_forecasts.append(predicted_values)
            all_actuals.append(actual_values)
            forecast_indices.append(np.arange(current_idx, current_idx + len(actual_values)))
            
            # Prepare for next iteration - expand training window
            if current_idx + forecast_horizon < len(X):
                new_train_end = current_idx + forecast_horizon
                X_train = X[:new_train_end]
                y_train = y[:new_train_end]
                
                # Retrain model with expanded data
                print(f"  Retraining on {len(X_train)} samples...")
                start_time = time.time()
                model.fit(X_train, y_train.ravel())
                training_time = time.time() - start_time
                training_times.append(training_time)
                print(f"  Training took {training_time:.2f} seconds")
        else:
            print(f"  Reached end of data, stopping")
            break
    
    # Calculate overall metrics
    all_actuals_flat = np.concatenate(all_actuals)
    all_forecasts_flat = np.concatenate(all_forecasts)
    
    mae = mean_absolute_error(all_actuals_flat, all_forecasts_flat)
    rmse = np.sqrt(mean_squared_error(all_actuals_flat, all_forecasts_flat))
    r2 = r2_score(all_actuals_flat, all_forecasts_flat)
    
    print("\nRolling origin forecast evaluation metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Average training time: {np.mean(training_times):.2f} seconds")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot the test data
    plt.plot(range(train_size, len(y)), y[train_size:], 'b-', alpha=0.5, label='Actual Values')
    
    # Plot all forecasts
    for idx, forecast, actual in zip(forecast_indices, all_forecasts, all_actuals):
        plt.plot(idx, forecast, 'r-', linewidth=2, alpha=0.7)
    
    plt.title('Rolling Origin Forecasts (Red) vs. Actual Values (Blue)')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('rolling_origin_forecasts.png')
    plt.show()
    
    return all_forecasts, all_actuals, forecast_indices, model

# Run the evaluation
forecasts, actuals, indices, final_model = rolling_origin_forecast(X, y)