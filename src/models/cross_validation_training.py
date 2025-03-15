"""
Cross-validation training pipeline with memory-efficient processing.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional, Union
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gc
import time
import json

# Import our data loading utilities
from data.data_loader import (
    get_cv_round_count,
    cv_round_generator,
    preprocess_data
)


def train_xgb_model(X_train: pd.DataFrame, 
                   y_train: pd.DataFrame, 
                   params: Optional[Dict] = None) -> xgb.XGBRegressor:
    """
    Train an XGBoost model with the given parameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        params: XGBoost parameters (optional)
        
    Returns:
        Trained XGBoost model
    """
    # Add validation
    if X_train.empty or y_train.empty:
        raise ValueError(f"Empty training data: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Default parameters if none provided
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 20,
            'random_state': 42
        }
    
    # Initialize model
    model = xgb.XGBRegressor(**params)
    
    # Convert y_train to numpy array more carefully
    # Try different conversion methods
    try:
        # First attempt: convert directly to numpy array
        y_array = y_train.values
        
        # Check if resulting array is empty
        if len(y_array) == 0:
            print("WARNING: y_train.values produced an empty array, trying alternative conversion")
            # Try another method
            y_array = np.array(y_train)
            
        # Ensure it's 1D array (ravel if needed)
        y_array = y_array.ravel()
        
        print(f"Final y_array shape: {y_array.shape}")
        
        if len(y_array) != len(X_train):
            raise ValueError(f"Length mismatch: X_train has {len(X_train)} rows but y_array has {len(y_array)} elements")
            
        # Train the model
        model.fit(X_train, y_array)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("\nDEBUG INFO:")
        print(f"y_train head: {y_train.head()}")
        print(f"y_train columns: {y_train.columns}")
        # Try a direct string access to the first column
        if len(y_train.columns) > 0:
            first_col = y_train.columns[0]
            print(f"Trying direct access to first column: {first_col}")
            y_direct = y_train[first_col].values
            print(f"Direct column access shape: {y_direct.shape}")
            model.fit(X_train, y_direct)
        else:
            raise ValueError("y_train has no columns")
    
    return model


def evaluate_model(model: xgb.XGBRegressor, 
                  X_test: pd.DataFrame, 
                  y_test: pd.DataFrame,
                  iterative_forecast: bool = False,
                  target_lag_idx: Optional[int] = None) -> Dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        iterative_forecast: Whether to use iterative forecasting
        target_lag_idx: Index of target lag feature for iterative forecasting
        
    Returns:
        Dictionary with evaluation metrics
    """
    # For single-step prediction
    if not iterative_forecast or target_lag_idx is None:
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'method': 'single-step'
        }
    
    # For iterative forecasting
    else:
        predictions = []
        for i in range(len(X_test)):
            # Get initial features for this sequence
            initial_features = X_test.iloc[i:i+1].values
            
            # Get actual sequence length
            actual_sequence = y_test.iloc[i:].values.ravel()
            horizon = len(actual_sequence)
            
            if horizon == 0:
                continue
                
            # Generate iterative forecast
            predicted_sequence = iterative_forecast_fn(
                model, 
                initial_features[0], 
                steps=horizon,
                target_lag_idx=target_lag_idx
            )
            
            predictions.append((i, actual_sequence, predicted_sequence))
        
        # Calculate metrics across all sequences
        all_actuals = np.concatenate([actual for _, actual, _ in predictions])
        all_preds = np.concatenate([pred for _, _, pred in predictions])
        
        mae = mean_absolute_error(all_actuals, all_preds)
        rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
        r2 = r2_score(all_actuals, all_preds)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'method': 'iterative'
        }


def iterative_forecast_fn(model, X_initial, steps=32, target_lag_idx=None):
    """
    Generate an iterative forecast.
    
    Args:
        model: Trained model
        X_initial: Initial feature vector
        steps: Number of steps to forecast
        target_lag_idx: Index of the target lag feature
        
    Returns:
        Array of predicted values
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


def run_cross_validation(params: Dict,
                        start_round: int = 0,
                        end_round: Optional[int] = None,
                        step: int = 1,
                        save_results: bool = True,
                        output_dir: str = './results',
                        model_type: str = 'xgboost',
                        store_predictions: bool = True):
    """
    Run cross-validation with the specified parameters and store predictions.
    
    Args:
        params: Model parameters
        start_round: First CV round to use
        end_round: Last CV round to use (None = all available)
        step: Step size for iterating through rounds
        save_results: Whether to save results to disk
        output_dir: Directory to save results
        model_type: Type of model to train ('xgboost' or other)
        store_predictions: Whether to store actual vs. predicted values
    """
    # Create output directory if it doesn't exist
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results collection
    results = {
        'params': params,
        'rounds': {},
        'summary': {},
        'predictions': {} if store_predictions else None
    }
    
    # Track metrics across all rounds
    all_metrics = {
        'mae': [],
        'rmse': [],
        'r2': []
    }
    
    # Identify total number of rounds
    total_rounds = get_cv_round_count() if end_round is None else end_round - start_round
    print(f"Running cross-validation on {total_rounds} rounds with step size {step}")
    
    # Use generator to process one round at a time (memory efficient)
    for round_num, X_train, y_train, X_test, y_test in cv_round_generator(start_round, end_round, step):
        print(f"\nProcessing round {round_num}...")
        round_start_time = time.time()
        
        # Store original test data for plotting
        if store_predictions:
            original_y_test = y_test.copy()
        
        # Preprocess data
        X_train, y_train, X_test, y_test = preprocess_data(
            X_train, y_train, X_test, y_test,
            scale_features=True,
            remove_timezone=True
        )
        
        # Train model
        print(f"Training model...")
        if model_type == 'xgboost':
            model = train_xgb_model(X_train, y_train, params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Evaluate on test set
        print(f"Evaluating model...")
        
        # Try to find target lag column for iterative forecasting
        target_lag_idx = None
        target_col = y_train.columns[0]
        for i, col in enumerate(X_train.columns):
            if target_col in col and "_lag_" in col:
                target_lag_idx = i
                print(f"Found target lag column: {col} at index {i}")
                break
        
        # Single-step evaluation
        single_step_metrics = evaluate_model(model, X_test, y_test, 
                                          iterative_forecast=False)
        
        # Store predictions if requested
        if store_predictions:
            y_pred = single_step_metrics['predictions']
            
            # Create DataFrame with actual and predicted values
            pred_df = pd.DataFrame({
                'actual': original_y_test.iloc[:, 0].values,
                'predicted': y_pred,
                'timestamp': original_y_test.index
            })
            
            # Store in results
            results['predictions'][round_num] = pred_df
            
            # Calculate additional metrics that might help explain poor R²
            variance_y = np.var(pred_df['actual'])
            mean_y = np.mean(pred_df['actual'])
            min_y = np.min(pred_df['actual'])
            max_y = np.max(pred_df['actual'])
            
            # Print info about test set distribution
            print(f"  Test set variance: {variance_y:.4f}")
            print(f"  Test set mean: {mean_y:.4f}")
            print(f"  Test set range: [{min_y:.4f}, {max_y:.4f}]")
            
            # Save prediction CSV
            if save_results:
                pred_df.to_csv(f"{output_dir}/round_{round_num}_predictions.csv", index=False)
            
            # Create and save plot
            if save_results:
                plt.figure(figsize=(12, 6))
                
                # Plot actual vs predicted values
                plt.subplot(1, 2, 1)
                plt.plot(pred_df['timestamp'], pred_df['actual'], 'b-', label='Actual')
                plt.plot(pred_df['timestamp'], pred_df['predicted'], 'r--', label='Predicted')
                plt.title(f'Round {round_num} - Actual vs Predicted')
                plt.ylabel('Price')
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                
                # Scatter plot to visualize correlation
                plt.subplot(1, 2, 2)
                plt.scatter(pred_df['actual'], pred_df['predicted'])
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title(f'Correlation Plot (R² = {single_step_metrics["r2"]:.4f})')
                min_val = min(pred_df['actual'].min(), pred_df['predicted'].min())
                max_val = max(pred_df['actual'].max(), pred_df['predicted'].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)  # Perfect prediction line
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/round_{round_num}_pred_vs_actual.png")
                plt.close()
        
        # Iterative forecasting evaluation (if target lag found)
        if target_lag_idx is not None:
            iterative_metrics = evaluate_model(model, X_test, y_test,
                                           iterative_forecast=True,
                                           target_lag_idx=target_lag_idx)
        else:
            iterative_metrics = None
            print("No target lag column found, skipping iterative forecasting")
        
        # Calculate round duration
        round_duration = time.time() - round_start_time
        
        # Store results for this round
        round_results = {
            'single_step': single_step_metrics,
            'iterative': iterative_metrics,
            'duration': round_duration
        }
        
        # Add to overall results
        results['rounds'][round_num] = round_results
        
        # Update metrics tracking
        all_metrics['mae'].append(single_step_metrics['mae'])
        all_metrics['rmse'].append(single_step_metrics['rmse'])
        all_metrics['r2'].append(single_step_metrics['r2'])
        
        # Print round summary
        print(f"Round {round_num} results:")
        print(f"  Single-step MAE: {single_step_metrics['mae']:.4f}")
        print(f"  Single-step RMSE: {single_step_metrics['rmse']:.4f}")
        print(f"  Single-step R²: {single_step_metrics['r2']:.4f}")
        
        if iterative_metrics:
            print(f"  Iterative MAE: {iterative_metrics['mae']:.4f}")
            print(f"  Iterative RMSE: {iterative_metrics['rmse']:.4f}")
            print(f"  Iterative R²: {iterative_metrics['r2']:.4f}")
        
        print(f"  Duration: {round_duration:.2f} seconds")
        
        # Save partial results after each round
        if save_results:
            timestamp = int(time.time())
            
            # Save round metrics (without the large prediction arrays)
            round_metrics = {
                'single_step': {
                    'mae': single_step_metrics['mae'],
                    'rmse': single_step_metrics['rmse'],
                    'r2': single_step_metrics['r2'],
                    'method': single_step_metrics['method']
                },
                'duration': round_duration
            }
            
            if iterative_metrics:
                round_metrics['iterative'] = {
                    'mae': iterative_metrics['mae'],
                    'rmse': iterative_metrics['rmse'],
                    'r2': iterative_metrics['r2'],
                    'method': iterative_metrics['method']
                }
            
            with open(f"{output_dir}/round_{round_num}_results_{timestamp}.json", 'w') as f:
                json.dump(round_metrics, f, indent=2)
            
            # Save plot for iterative forecasting
            if iterative_metrics and 'predictions' in iterative_metrics:
                try:
                    # Get a sample prediction for visualization
                    if len(iterative_metrics['predictions']) > 0:
                        idx, actual, predicted = iterative_metrics['predictions'][0]
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(actual, 'b-', label='Actual')
                        plt.plot(predicted, 'r--', label='Predicted')
                        plt.title(f'Round {round_num} - Iterative Forecast')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(f"{output_dir}/round_{round_num}_iterative_forecast.png")
                        plt.close()
                except Exception as e:
                    print(f"Error saving plot: {str(e)}")
        
        # Clean up memory
        del X_train, y_train, X_test, y_test, model
        if store_predictions and 'pred_df' in locals():
            del pred_df
        gc.collect()
    
    # Calculate summary statistics
    results['summary'] = {
        'avg_mae': np.mean(all_metrics['mae']),
        'avg_rmse': np.mean(all_metrics['rmse']),
        'avg_r2': np.mean(all_metrics['r2']),
        'std_mae': np.std(all_metrics['mae']),
        'std_rmse': np.std(all_metrics['rmse']),
        'std_r2': np.std(all_metrics['r2']),
        'min_mae': np.min(all_metrics['mae']),
        'min_rmse': np.min(all_metrics['rmse']),
        'max_r2': np.max(all_metrics['r2']),
        'median_r2': np.median(all_metrics['r2'])  # Added median as it's more robust to outliers
    }
    
    # Print overall summary
    print("\nCross-validation summary:")
    print(f"Avg MAE: {results['summary']['avg_mae']:.4f} ± {results['summary']['std_mae']:.4f}")
    print(f"Avg RMSE: {results['summary']['avg_rmse']:.4f} ± {results['summary']['std_rmse']:.4f}")
    print(f"Avg R²: {results['summary']['avg_r2']:.4f} ± {results['summary']['std_r2']:.4f}")
    print(f"Median R²: {results['summary']['median_r2']:.4f}")
    print(f"Max R²: {results['summary']['max_r2']:.4f}")
    
    # Generate combined visualization
    if store_predictions and save_results:
        create_combined_visualization(results, output_dir)
    
    # Save final summary
    if save_results:
        # Create a NumPy-safe JSON serializer
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
                
        with open(f"{output_dir}/cv_summary_{int(time.time())}.json", 'w') as f:
            # Filter out predictions from the results as they're too large
            summary_results = {k: v for k, v in results.items() if k != 'predictions'}
            json.dump(summary_results, f, indent=2, default=convert_numpy_types)
    
    return results


def create_combined_visualization(results, output_dir):
    """
    Create a combined visualization of actual vs predicted values across all rounds.
    
    Args:
        results: Results dictionary with predictions
        output_dir: Directory to save visualization
    """
    try:
        # Collect all r2 values
        r2_values = []
        round_nums = []
        
        for round_num, round_data in results['rounds'].items():
            if 'single_step' in round_data and 'r2' in round_data['single_step']:
                r2_values.append(round_data['single_step']['r2'])
                round_nums.append(round_num)
        
        # Sort round numbers
        sorted_indices = np.argsort(round_nums)
        sorted_rounds = [round_nums[i] for i in sorted_indices]
        sorted_r2 = [r2_values[i] for i in sorted_indices]
        
        # Plot R² values across rounds
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.bar(sorted_rounds, sorted_r2)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Round Number')
        plt.ylabel('R²')
        plt.title('R² Values Across Cross-Validation Rounds')
        plt.grid(True, axis='y')
        
        # Plot R² distribution (histogram)
        plt.subplot(2, 1, 2)
        plt.hist(sorted_r2, bins=20)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('R² Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of R² Values')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/r2_distribution.png")
        plt.close()
        
        # Create a combined scatter plot for sample of rounds
        max_rounds_to_plot = 9  # 3x3 grid
        sample_rounds = sorted(results['predictions'].keys())
        
        if len(sample_rounds) > max_rounds_to_plot:
            # Select rounds evenly spaced
            step = len(sample_rounds) // max_rounds_to_plot
            sample_rounds = sample_rounds[::step][:max_rounds_to_plot]
        
        if sample_rounds:
            rows = int(np.ceil(np.sqrt(len(sample_rounds))))
            cols = int(np.ceil(len(sample_rounds) / rows))
            
            plt.figure(figsize=(15, 10))
            
            for i, round_num in enumerate(sample_rounds):
                if round_num in results['predictions']:
                    pred_df = results['predictions'][round_num]
                    r2 = results['rounds'][round_num]['single_step']['r2']
                    
                    plt.subplot(rows, cols, i+1)
                    plt.scatter(pred_df['actual'], pred_df['predicted'], alpha=0.7)
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')
                    plt.title(f'Round {round_num} (R²={r2:.4f})')
                    
                    # Add perfect prediction line
                    min_val = min(pred_df['actual'].min(), pred_df['predicted'].min())
                    max_val = max(pred_df['actual'].max(), pred_df['predicted'].max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)
                    plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/combined_correlation_plots.png")
            plt.close()
            
            # Create time series plots for sample rounds
            plt.figure(figsize=(15, 10))
            
            for i, round_num in enumerate(sample_rounds):
                if round_num in results['predictions']:
                    pred_df = results['predictions'][round_num]
                    mae = results['rounds'][round_num]['single_step']['mae']
                    
                    plt.subplot(rows, cols, i+1)
                    plt.plot(pred_df['timestamp'], pred_df['actual'], 'b-', label='Actual')
                    plt.plot(pred_df['timestamp'], pred_df['predicted'], 'r--', label='Predicted')
                    plt.title(f'Round {round_num} (MAE={mae:.4f})')
                    if i == 0:
                        plt.legend()
                    plt.xticks(rotation=45)
                    plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/combined_time_series_plots.png")
            plt.close()
    
    except Exception as e:
        print(f"Error creating combined visualization: {str(e)}")

if __name__ == "__main__":
    # Example usage
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'max_depth': 20,
        'random_state': 42
    }
    
    # Run cross-validation on a subset of rounds
    results = run_cross_validation(
        params=xgb_params,
        start_round=0,
        end_round=10,  # Only process first 10 rounds for this example
        step=1,
        save_results=True,
        output_dir='./results/xgb_cv_run'
    )
