#!/usr/bin/env python3
"""
ebm_xgboost_cv_all_targets.py - Apply a stacked EBM+XGBoost model to multiple target areas

This script applies a stacked model to datasets across different areas:
1. An Explainable Boosting Machine (EBM) is trained as the base model
2. XGBoost is trained on the residuals of the EBM predictions
3. Final predictions are the sum of both models' outputs

The script follows the same structure as run_cv_all_targets.py but with this custom stacked model.
Performance optimized version to reduce training time.
"""

import os
import argparse
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.linear_model import LinearRegression
import multiprocessing
from functools import partial
import warnings

# Suppress warnings to reduce output clutter
warnings.filterwarnings('ignore')

def get_available_targets(organized_dir="./src/data/csv", areas=["no1", "no2", "no3", "no4", "no5"]):
    """
    Get list of available targets from the organized directory structure.
    
    Args:
        organized_dir: Base directory for organized files
        areas: List of area directories to check
        
    Returns:
        List of available targets (directory names)
    """
    targets = []
    
    # Look for subdirectories that match the specified areas
    for item in os.listdir(organized_dir):
        if item in areas:
            targets.append(item)
    
    return targets

def load_cv_round(cv_round, target_dir, x_files_dir, target_index=0):
    """
    Load a specific cross-validation round's data for a target.
    
    Args:
        cv_round: The cross-validation round number to load
        target_dir: Directory containing target-specific files
        x_files_dir: Directory containing X files
        target_index: Index (0-based) for which target column to return if multiple targets exist
        
    Returns:
        Tuple containing (X_train, y_train, X_test, y_test) as pandas DataFrames or Series.
        y_train and y_test will be the selected target column.
    """
    try:
        # Load X data
        X_train = pd.read_csv(f"{x_files_dir}/X_train_{cv_round}.csv", index_col=0)
        X_test = pd.read_csv(f"{x_files_dir}/X_test_{cv_round}.csv", index_col=0)
        
        # Load y data from target-specific directory
        y_train = pd.read_csv(f"{target_dir}/y_train_{cv_round}.csv", index_col=0)
        y_test = pd.read_csv(f"{target_dir}/y_test_{cv_round}.csv", index_col=0)
        
        # Convert indices to datetime with utc=True to avoid warnings
        X_train.index = pd.to_datetime(X_train.index, utc=True)
        X_test.index = pd.to_datetime(X_test.index, utc=True)
        y_train.index = pd.to_datetime(y_train.index, utc=True)
        y_test.index = pd.to_datetime(y_test.index, utc=True)
        
        # If y_train is a DataFrame with multiple columns, select the column based on target_index.
        if isinstance(y_train, pd.DataFrame):
            if y_train.shape[1] > 1:
                y_train = y_train.iloc[:, target_index]
                y_test = y_test.iloc[:, target_index]
            else:
                # If there is only one column, simply squeeze to a Series.
                y_train = y_train.squeeze()
                y_test = y_test.squeeze()
        
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"Error loading CV round {cv_round}: {e}")
        raise

def get_cv_round_count(target_dir):
    """
    Get the total number of cross-validation rounds available for a target.
    
    Args:
        target_dir: Directory containing target-specific files
        
    Returns:
        int: The number of cross-validation rounds
    """
    y_test_files = [f for f in os.listdir(target_dir) if f.startswith('y_test_')]
    
    # Extract round numbers from filenames
    round_numbers = []
    for f in y_test_files:
        try:
            round_num = int(f.split('_')[-1].split('.')[0])
            round_numbers.append(round_num)
        except (ValueError, IndexError):
            continue
    
    if not round_numbers:
        raise FileNotFoundError(f"No cross-validation files found in {target_dir}")
    
    return max(round_numbers) + 1  # +1 because we count from 0

def train_and_evaluate_stacked_model(X_train, y_train, X_test, y_test, ebm_params=None, xgb_params=None, fast_mode=True):
    """
    Train and evaluate a stacked model (EBM + XGBoost on residuals) with proper data cleaning.
    Fast mode option uses simplified EBM parameters for speed.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        ebm_params: Parameters for the EBM model
        xgb_params: Parameters for the XGBoost model
        fast_mode: If True, use simplified EBM for much faster training
        
    Returns:
        Dictionary with evaluation metrics, models and predictions
    """
    # Default parameters if none provided
    if ebm_params is None:
        if fast_mode:
            # Fast mode parameters - minimal interactions, fewer bins
            ebm_params = {
                'max_bins': 64,             # Reduced from 256
                'max_interaction_bins': 16, # Reduced from 32
                'interactions': 0,          # No interactions by default for speed
                'learning_rate': 0.05,      # Increased from 0.01
                'min_samples_leaf': 10,     # Increased from 5
                'random_state': 42,
                'outer_bags': 4,            # Reduced ensemble size
                'inner_bags': 0,            # No inner bags for speed
                'max_rounds': 1000,         # Limit iterations
                'early_stopping_rounds': 50 # Add early stopping
            }
        else:
            # Standard mode parameters
            ebm_params = {
                'max_bins': 256,
                'max_interaction_bins': 32,
                'interactions': 10,
                'learning_rate': 0.01,
                'min_samples_leaf': 5,
                'random_state': 42
            }
    
    if xgb_params is None:
        xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,            # Reduced from 500
            'learning_rate': 0.1,           # Increased from 0.05
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'random_state': 42,
            'early_stopping_rounds': 10     # Add early stopping
        }
    
    # Handle whether y_train/y_test are Series or DataFrame
    if isinstance(y_train, pd.DataFrame):
        y_train_vals = y_train.iloc[:, 0].values
    else:  # It's a Series
        y_train_vals = y_train.values
        
    if isinstance(y_test, pd.DataFrame):
        y_test_vals = y_test.iloc[:, 0].values
    else:  # It's a Series
        y_test_vals = y_test.values
    
    # Check for and handle NaN or infinity values in training data
    is_valid_train = ~(np.isnan(y_train_vals) | np.isinf(y_train_vals) | (np.abs(y_train_vals) > 1e10))
    
    if not np.all(is_valid_train):
        num_invalid = np.sum(~is_valid_train)
        print(f"  Warning: Found {num_invalid} invalid values in training labels. Removing these rows.")
        
        # Filter out invalid values
        valid_indices = np.where(is_valid_train)[0]
        X_train_clean = X_train.iloc[valid_indices]
        y_train_vals_clean = y_train_vals[valid_indices]
    else:
        X_train_clean = X_train
        y_train_vals_clean = y_train_vals
    
    # Check for and handle NaN or infinity values in test data
    is_valid_test = ~(np.isnan(y_test_vals) | np.isinf(y_test_vals) | (np.abs(y_test_vals) > 1e10))
    
    if not np.all(is_valid_test):
        num_invalid = np.sum(~is_valid_test)
        print(f"  Warning: Found {num_invalid} invalid values in test labels. Removing these rows.")
        
        # Filter out invalid values
        valid_indices = np.where(is_valid_test)[0]
        X_test_clean = X_test.iloc[valid_indices]
        y_test_vals_clean = y_test_vals[valid_indices]
    else:
        X_test_clean = X_test
        y_test_vals_clean = y_test_vals
    
    # Check if we have enough data left after cleaning
    if len(y_train_vals_clean) < 10 or len(y_test_vals_clean) < 10:
        print("  Warning: Not enough valid data points after cleaning. Skipping this round.")
        return {
            'ebm_model': None,
            'xgb_model': None,
            'metrics': {
                'mae': float('nan'),
                'rmse': float('nan')
            },
            'predictions': pd.DataFrame(columns=['actual', 'predicted', 'ebm_pred', 'xgb_pred', 'timestamp']),
            'rows_removed': {
                'train': np.sum(~is_valid_train),
                'test': np.sum(~is_valid_test)
            }
        }
    
    try:
        # Start timing the model training
        start_time = time.time()
        
        # Step 1: Train EBM model
        print("  Training EBM model...")
        ebm_model = ExplainableBoostingRegressor(**ebm_params)
        ebm_model.fit(X_train_clean, y_train_vals_clean)
        
        # Get EBM predictions on train and test data
        ebm_train_preds = ebm_model.predict(X_train_clean)
        ebm_test_preds = ebm_model.predict(X_test_clean)
        
        # Step 2: Calculate residuals for training data
        train_residuals = y_train_vals_clean - ebm_train_preds
        
        # Step 3: Train XGBoost on the residuals
        print("  Training XGBoost model on residuals...")
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_clean, train_residuals, 
                      eval_set=[(X_train_clean, train_residuals)],
                      verbose=0)
        
        # Get XGBoost predictions on test data (predicting residuals)
        xgb_test_preds = xgb_model.predict(X_test_clean)
        
        # Step 4: Final predictions = EBM predictions + XGBoost predictions
        final_predictions = ebm_test_preds + xgb_test_preds
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_vals_clean, final_predictions)
        rmse = np.sqrt(mean_squared_error(y_test_vals_clean, final_predictions))
        
        # Record training time
        training_time = time.time() - start_time
        print(f"  Model training completed in {training_time:.2f} seconds")
        
        # Create DataFrame with predictions
        pred_df = pd.DataFrame({
            'actual': y_test_vals_clean,
            'predicted': final_predictions,
            'ebm_pred': ebm_test_preds,
            'xgb_pred': xgb_test_preds,
            'timestamp': X_test_clean.index,
            'round': np.full(len(y_test_vals_clean), -1)  # Will be filled with round number by caller
        })
        
        return {
            'ebm_model': ebm_model,
            'xgb_model': xgb_model,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'training_time': training_time
            },
            'predictions': pred_df,
            'rows_removed': {
                'train': np.sum(~is_valid_train),
                'test': np.sum(~is_valid_test)
            }
        }
    except Exception as e:
        print(f"  Error in training or evaluation: {str(e)}")
        return {
            'ebm_model': None,
            'xgb_model': None,
            'metrics': {
                'mae': float('nan'),
                'rmse': float('nan')
            },
            'predictions': pd.DataFrame(columns=['actual', 'predicted', 'ebm_pred', 'xgb_pred', 'timestamp']),
            'rows_removed': {
                'train': np.sum(~is_valid_train),
                'test': np.sum(~is_valid_test)
            },
            'error': str(e)
        }

def process_single_round(round_num, target, target_dir, x_files_dir, target_index, 
                        ebm_params, xgb_params, index_output_dir, fast_mode):
    """
    Process a single CV round - helper function for parallelization.
    
    Returns:
        Dictionary with round results or None if error
    """
    try:
        print(f"\nProcessing round {round_num}...")
        
        # Load data for this round
        X_train, y_train, X_test, y_test = load_cv_round(
            cv_round=round_num,
            target_dir=target_dir,
            x_files_dir=x_files_dir,
            target_index=target_index
        )

        # Train and evaluate stacked model
        result = train_and_evaluate_stacked_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            ebm_params=ebm_params,
            xgb_params=xgb_params,
            fast_mode=fast_mode
        )

        # Check if we skipped this round due to not enough valid data or errors
        if result['ebm_model'] is None or result['xgb_model'] is None:
            print(f"  Skipping round {round_num} due to model training errors or insufficient valid data.")
            return None
        
        # Add round number to predictions
        result['predictions']['round'] = round_num

        # Print round results
        print(f"  MAE: {result['metrics']['mae']:.4f}")
        print(f"  RMSE: {result['metrics']['rmse']:.4f}")
        if 'training_time' in result['metrics']:
            print(f"  Training time: {result['metrics']['training_time']:.2f} seconds")

        if 'rows_removed' in result:
            print(f"  Rows removed: {result['rows_removed']['train']} train, {result['rows_removed']['test']} test")

        # Save predictions to the shared index output directory
        pred_df = result['predictions']
        pred_df.to_csv(os.path.join(index_output_dir, f"round_{round_num}_predictions.csv"), index=False)

        # Create basic plot (we'll generate detailed plots later)
        plt.figure(figsize=(12, 6))
        plt.plot(pred_df['timestamp'], pred_df['actual'], 'b-', label='Actual')
        plt.plot(pred_df['timestamp'], pred_df['predicted'], 'r--', label='Predicted')
        plt.title(f'Round {round_num} - Actual vs Predicted')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(index_output_dir, f"round_{round_num}_plot.png"))
        plt.close()

        return {
            'round_num': round_num,
            'metrics': result['metrics'],
            'predictions': result['predictions']
        }
    except Exception as e:
        print(f"Error processing round {round_num}: {str(e)}")
        return None

def sample_rounds(start_round, end_round, step, max_rounds=10):
    """
    Sample a limited number of rounds evenly from the available range
    to reduce computational load.
    """
    all_rounds = list(range(start_round, end_round, step))
    
    if len(all_rounds) <= max_rounds:
        return all_rounds
    
    # Sample rounds evenly across the range
    indices = np.linspace(0, len(all_rounds) - 1, max_rounds, dtype=int)
    sampled_rounds = [all_rounds[i] for i in indices]
    
    return sampled_rounds

def run_cv_for_target_stacked(target, start_round=0, end_round=None, step=1,
                            ebm_params=None, xgb_params=None, output_dir=None,
                            organized_dir="./src/data/csv", target_index=0,
                            parallel=True, max_workers=None, fast_mode=True,
                            sample_size=None):
    """
    Run cross-validation for a specific target with stacked model (EBM+XGBoost) and overall R² calculation.
    Performance optimized with optional parallelization and sampling.
    
    Args:
        target: Target directory name
        start_round: First CV round to process
        end_round: Last CV round to process (None = all available)
        step: Step size for processing rounds
        ebm_params: Parameters for the EBM model
        xgb_params: Parameters for the XGBoost model
        output_dir: Directory to save results
        organized_dir: Base directory for organized files
        target_index: Index of target variable (0 or 1) for regulation up/down
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers (None = auto)
        fast_mode: Use simplified EBM parameters for faster training
        sample_size: Maximum number of rounds to process (None = process all specified rounds)
        
    Returns:
        Dictionary with results summary
    """
    # Set up paths
    target_dir = os.path.join(organized_dir, target)
    x_files_dir = os.path.join(organized_dir, target)
    
    # Check if target directory exists
    if not os.path.isdir(target_dir):
        raise ValueError(f"Target directory not found: {target_dir}")
    
    # Set up output directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Get total number of rounds
    total_rounds = get_cv_round_count(target_dir)
    
    # Set default end_round if not provided
    if end_round is None:
        end_round = total_rounds
    else:
        end_round = min(end_round, total_rounds)
    
    # Create a unique directory for the run
    if output_dir is None:
        base_output_dir = f"./results/stacked"
    else:
        base_output_dir = output_dir
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Determine the rounds to process
    if sample_size is not None:
        rounds_to_process = sample_rounds(start_round, end_round, step, sample_size)
        print(f"Sampling {len(rounds_to_process)} rounds from {(end_round-start_round)//step} available")
    else:
        rounds_to_process = list(range(start_round, end_round, step))
        print(f"Processing all {len(rounds_to_process)} available rounds")
    
    # Create index-specific output directory with timestamp
    index_output_dir = os.path.join(base_output_dir, f"{target}_ind_{target_index}_{timestamp}")
    os.makedirs(index_output_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'target': target,
        'start_round': start_round,
        'end_round': end_round,
        'step': step,
        'ebm_params': ebm_params,
        'xgb_params': xgb_params,
        'timestamp': timestamp,
        'target_index': target_index,
        'fast_mode': fast_mode,
        'sampled_rounds': rounds_to_process
    }
    
    with open(os.path.join(index_output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nRunning stacked model (EBM+XGBoost) cross-validation for target: {target}, index: {target_index}")
    print(f"Processing {len(rounds_to_process)} rounds out of {total_rounds} total rounds")
    print(f"Fast mode: {fast_mode}, Parallel: {parallel}")
    if ebm_params:
        print(f"EBM interactions: {ebm_params.get('interactions', 'Not specified')}")
    
    start_time_all = time.time()
    
    # Process rounds
    if parallel and len(rounds_to_process) > 1:
        # Use multiprocessing for parallelism
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(rounds_to_process))
        
        print(f"Running with {max_workers} parallel workers")
        
        # Create a partial function with fixed arguments
        process_func = partial(
            process_single_round,
            target=target,
            target_dir=target_dir,
            x_files_dir=x_files_dir,
            target_index=target_index,
            ebm_params=ebm_params,
            xgb_params=xgb_params,
            index_output_dir=index_output_dir,
            fast_mode=fast_mode
        )
        
        # Run in parallel
        with multiprocessing.Pool(max_workers) as pool:
            results = pool.map(process_func, rounds_to_process)
        
        # Filter out None results (failed rounds)
        results = [r for r in results if r is not None]
        
        # Process successful results
        round_results = {}
        all_predictions = []
        
        for r in results:
            round_num = r['round_num']
            round_results[round_num] = r['metrics']
            all_predictions.append(r['predictions'])
        
        successful_rounds = len(results)
        failed_rounds = len(rounds_to_process) - successful_rounds
        
    else:
        # Process sequentially
        round_results = {}
        all_predictions = []
        successful_rounds = 0
        failed_rounds = 0
        
        for round_num in rounds_to_process:
            # Use the shared index output dir
            result = process_single_round(
                round_num,
                target,
                target_dir,
                x_files_dir,
                target_index,
                ebm_params,
                xgb_params,
                index_output_dir,
                fast_mode
            )
            
            if result is not None:
                round_results[round_num] = result['metrics']
                all_predictions.append(result['predictions'])
                successful_rounds += 1
            else:
                failed_rounds += 1
    
    total_time = time.time() - start_time_all
    avg_time_per_round = total_time / max(1, successful_rounds)
    
    print(f"\nTotal processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per round: {avg_time_per_round:.2f} seconds")
    
    # Check if we have any successful rounds
    if not round_results:
        print(f"No successful rounds for target {target}. All {failed_rounds} rounds failed.")

        # Save a minimal summary
        summary = {
            'processed_rounds': failed_rounds + successful_rounds,
            'successful_rounds': successful_rounds,
            'failed_rounds': failed_rounds,
            'error': "All rounds failed",
            'total_time': total_time
        }

        with open(os.path.join(index_output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        return summary

    # Combine all predictions for overall R² calculation
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)

        # Calculate overall metrics on all predictions combined
        overall_actual = combined_predictions['actual'].values
        overall_predicted = combined_predictions['predicted'].values
        overall_ebm_pred = combined_predictions['ebm_pred'].values
        
        # Handle potential division by zero in R² calculation
        if np.var(overall_actual) > 0:
            overall_r2 = r2_score(overall_actual, overall_predicted)
            overall_r2_ebm = r2_score(overall_actual, overall_ebm_pred)
        else:
            overall_r2 = float('nan')
            overall_r2_ebm = float('nan')
            print("Warning: Zero variance in combined actual values, overall R² is undefined")

        # Save combined predictions
        combined_predictions.to_csv(os.path.join(index_output_dir, "all_predictions.csv"), index=False)

        # Create overall correlation plot
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Overall Correlation - Final Prediction
        plt.subplot(2, 2, 1)
        plt.scatter(overall_actual, overall_predicted, alpha=0.5)
        plt.xlabel('Actual Values')
        plt.ylabel('Final Predicted Values')

        if not np.isnan(overall_r2):
            plt.title(f'Overall Final Prediction Correlation (R² = {overall_r2:.4f})')
        else:
            plt.title('Overall Final Prediction Correlation (R² undefined)')

        min_val = min(overall_actual.min(), overall_predicted.min())
        max_val = max(overall_actual.max(), overall_predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)
        plt.grid(True)

        # Plot 2: Overall Correlation - EBM Only
        plt.subplot(2, 2, 2)
        plt.scatter(overall_actual, overall_ebm_pred, alpha=0.5)
        plt.xlabel('Actual Values')
        plt.ylabel('EBM Predicted Values')

        if not np.isnan(overall_r2_ebm):
            plt.title(f'Overall EBM Prediction Correlation (R² = {overall_r2_ebm:.4f})')
        else:
            plt.title('Overall EBM Prediction Correlation (R² undefined)')

        plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)
        plt.grid(True)

        # Plot 3: Residual Analysis
        plt.subplot(2, 2, 3)
        final_residuals = overall_actual - overall_predicted
        plt.hist(final_residuals, bins=50)
        plt.title(f'Final Residuals Distribution (Mean: {np.mean(final_residuals):.4f})')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Plot 4: Model Contribution Comparison
        plt.subplot(2, 2, 4)
        ebm_r2 = overall_r2_ebm
        combined_r2 = overall_r2
        improvement = combined_r2 - ebm_r2
        
        bars = ['EBM Only', 'EBM+XGBoost', 'Improvement']
        values = [ebm_r2, combined_r2, improvement]
        colors = ['blue', 'green', 'red']
        
        plt.bar(bars, values, color=colors)
        plt.title('Model R² Comparison')
        plt.ylabel('R² Score')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.grid(axis='y')

        # Add summary statistics to plot
        overall_mae = mean_absolute_error(overall_actual, overall_predicted)
        overall_rmse = np.sqrt(mean_squared_error(overall_actual, overall_predicted))
        overall_mae_ebm = mean_absolute_error(overall_actual, overall_ebm_pred)
        overall_rmse_ebm = np.sqrt(mean_squared_error(overall_actual, overall_ebm_pred))

        plt.tight_layout()
        plt.savefig(os.path.join(index_output_dir, "overall_correlation.png"))
        plt.close()
    else:
        overall_r2 = float('nan')
        overall_r2_ebm = float('nan')
        overall_mae = float('nan')
        overall_rmse = float('nan')
        overall_mae_ebm = float('nan')
        overall_rmse_ebm = float('nan')

    # Extract MAE and RMSE values
    all_metrics = {
        'mae': [metrics['mae'] for metrics in round_results.values() if 'mae' in metrics],
        'rmse': [metrics['rmse'] for metrics in round_results.values() if 'rmse' in metrics],
        'training_time': [metrics.get('training_time', 0) for metrics in round_results.values()]
    }

    # Calculate summary statistics
    summary = {
        'avg_mae': np.mean(all_metrics['mae']) if all_metrics['mae'] else float('nan'),
        'avg_rmse': np.mean(all_metrics['rmse']) if all_metrics['rmse'] else float('nan'),
        'std_mae': np.std(all_metrics['mae']) if all_metrics['mae'] else float('nan'),
        'std_rmse': np.std(all_metrics['rmse']) if all_metrics['rmse'] else float('nan'),
        'min_mae': np.min(all_metrics['mae']) if all_metrics['mae'] else float('nan'),
        'avg_training_time': np.mean(all_metrics['training_time']) if all_metrics['training_time'] else float('nan'),
        'overall_r2': overall_r2,
        'overall_r2_ebm': overall_r2_ebm,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'overall_mae_ebm': overall_mae_ebm,
        'overall_rmse_ebm': overall_rmse_ebm,
        'processed_rounds': failed_rounds + successful_rounds,
        'successful_rounds': successful_rounds,
        'failed_rounds': failed_rounds,
        'total_time': total_time,
        'estimated_full_time': total_time * ((end_round - start_round) / step) / len(rounds_to_process) 
                                if rounds_to_process else float('nan')
    }

    # Print summary
    print("\nCross-validation summary:")
    print(f"Processed {summary['processed_rounds']} rounds ({summary['successful_rounds']} successful, {summary['failed_rounds']} failed)")
    print(f"Avg MAE: {summary['avg_mae']:.4f} ± {summary['std_mae']:.4f}")
    print(f"Avg RMSE: {summary['avg_rmse']:.4f} ± {summary['std_rmse']:.4f}")
    print(f"Overall R² (calculated on all predictions): {summary['overall_r2']:.4f}")
    print(f"Overall MAE: {summary['overall_mae']:.4f}")
    print(f"Overall RMSE: {summary['overall_rmse']:.4f}")
    print(f"EBM Base Model Overall R²: {summary['overall_r2_ebm']:.4f}")
    print(f"Average training time per round: {summary['avg_training_time']:.2f} seconds")
    print(f"Total processing time: {summary['total_time']:.2f} seconds ({summary['total_time']/60:.2f} minutes)")
    
    if sample_size is not None:
        print(f"Estimated time for all {(end_round - start_round) // step} rounds: {summary['estimated_full_time']/60:.2f} minutes")

    # Create MAE/RMSE summary plots
    if round_results:
        plt.figure(figsize=(12, 5))

        # Plot MAE across rounds
        plt.subplot(1, 2, 1)
        rounds = list(round_results.keys())
        mae_values = [round_results[r]['mae'] for r in rounds if 'mae' in round_results[r]]
        if mae_values:
            plt.bar(rounds, mae_values)
            plt.axhline(y=overall_mae, color='r', linestyle='-', label=f'Overall MAE: {overall_mae:.4f}')
            plt.xlabel('Round Number')
            plt.ylabel('MAE')
            plt.title('MAE Values Across Rounds')
            plt.grid(True, axis='y')
            plt.legend()

        # Plot RMSE across rounds
        plt.subplot(1, 2, 2)
        rmse_values = [round_results[r]['rmse'] for r in rounds if 'rmse' in round_results[r]]
        if rmse_values:
            plt.bar(rounds, rmse_values)
            plt.axhline(y=overall_rmse, color='r', linestyle='-', label=f'Overall RMSE: {overall_rmse:.4f}')
            plt.xlabel('Round Number')
            plt.ylabel('RMSE')
            plt.title('RMSE Values Across Rounds')
            plt.grid(True, axis='y')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(index_output_dir, "metrics_summary.png"))
        plt.close()

    # Save summary to file
    with open(os.path.join(index_output_dir, 'summary.json'), 'w') as f:
        # Convert numpy values to regular Python types for JSON serialization
        summary_json = {k: float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v) else 
                          None if isinstance(v, (np.floating, float)) and np.isnan(v) else v 
                          for k, v in summary.items()}
        json.dump(summary_json, f, indent=4)

    # Save individual round results
    with open(os.path.join(index_output_dir, 'round_results.json'), 'w') as f:
        # Convert to regular Python types
        round_results_json = {}
        for round_num, metrics in round_results.items():
            round_results_json[str(round_num)] = {
                k: float(v) if isinstance(v, np.floating) else v 
                for k, v in metrics.items()
            }
        json.dump(round_results_json, f, indent=4)

    return {
        'summary': summary,
        'round_results': round_results
    }

def main():
    parser = argparse.ArgumentParser(description='Run cross-validation with stacked EBM+XGBoost model and overall R² calculation')
    
    parser.add_argument('--targets', type=str, default=None,
                       help='Target to process, comma-separated (default: show available targets)')
    
    parser.add_argument('--start', type=int, default=0,
                       help='First CV round to process (default: 0)')
    
    parser.add_argument('--end', type=int, default=None,
                       help='Last CV round to process (default: all available)')
    
    parser.add_argument('--step', type=int, default=1,
                       help='Step size for processing rounds (default: 1)')
    
    parser.add_argument('--output', type=str, default='./results/stacked',
                       help='Base directory to save results (default: ./results/stacked)')
    
    parser.add_argument('--organized-dir', type=str, default='./src/data/csv',
                       help='Base directory for organized files (default: ./src/data/csv)')
    
    parser.add_argument('--list', action='store_true',
                       help='List available targets and exit')
    
    parser.add_argument('--target-index', type=int, default=None,
                        help='Index of target to process (0 or 1, default: run both)')
    
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: auto)')
    
    parser.add_argument('--no-fast-mode', action='store_true',
                        help='Disable fast mode (use full EBM parameters)')
    
    parser.add_argument('--sample-size', type=int, default=0,
                        help='Number of rounds to sample for processing (default: 0 = process all rounds)')
    
    # EBM parameters
    parser.add_argument('--ebm-interactions', type=int, default=0,
                       help='Number of interactions for EBM (default: 0)')
    
    parser.add_argument('--ebm-bins', type=int, default=64,
                       help='Number of bins for EBM (default: 64)')
    
    # XGBoost parameters
    parser.add_argument('--xgb-n-estimators', type=int, default=100,
                       help='Number of estimators for XGBoost (default: 100)')
    
    parser.add_argument('--xgb-learning-rate', type=float, default=0.1,
                       help='Learning rate for XGBoost (default: 0.1)')
    
    args = parser.parse_args()
    
    # Configure EBM parameters
    ebm_params = {
        'max_bins': args.ebm_bins,
        'max_interaction_bins': max(16, args.ebm_bins // 4),
        'interactions': args.ebm_interactions,
        'learning_rate': 0.05,
        'min_samples_leaf': 10,
        'random_state': 42,
        'outer_bags': 4,
        'inner_bags': 0,
        'max_rounds': 1000,
        'early_stopping_rounds': 50
    }
    
    # Configure XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': args.xgb_n_estimators,
        'learning_rate': args.xgb_learning_rate,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'random_state': 42,
        'early_stopping_rounds': 10
    }
    
    # Get available targets
    available_targets = get_available_targets(args.organized_dir)
    
    if args.list or not available_targets:
        print("\nAvailable targets:")
        for target in available_targets:
            target_dir = os.path.join(args.organized_dir, target)
            try:
                num_rounds = get_cv_round_count(target_dir)
                print(f"  {target} ({num_rounds} rounds)")
            except Exception as e:
                print(f"  {target} (Error: {str(e)})")
        return
    
    # Determine targets to process
    target_list = available_targets if args.targets is None else args.targets.split(',')
    
    # Process each target
    for target in target_list:
        if target not in available_targets:
            print(f"Target '{target}' not found in available targets.")
            continue
        
        try:
            print(f"\n{'='*50}")
            print(f"Processing target: {target}")
            print(f"{'='*50}")
            
            # Determine target indices to process
            if args.target_index is not None:
                target_indices = [args.target_index]
            else:
                target_indices = [0, 1]  # Process both up and down regulation
            
            for target_index in target_indices:
                # Create output directory
                os.makedirs(args.output, exist_ok=True)
                
                # Determine sample size (0 means process all - this is the default)
                sample_size = None if args.sample_size == 0 else args.sample_size
                
                run_cv_for_target_stacked(
                    target=target,
                    start_round=args.start,
                    end_round=args.end,
                    step=args.step,
                    ebm_params=ebm_params,
                    xgb_params=xgb_params,
                    output_dir=args.output,
                    organized_dir=args.organized_dir,
                    target_index=target_index,
                    parallel=not args.no_parallel,
                    max_workers=args.max_workers,
                    fast_mode=not args.no_fast_mode,
                    sample_size=sample_size
                )
        except Exception as e:
            print(f"Error processing target {target}: {str(e)}")

if __name__ == "__main__":
    # Set a higher recursion limit for complex models
    import sys
    sys.setrecursionlimit(10000)
    
    main()