#!/usr/bin/env python3
"""
ebm_xgboost_optimize.py - Sequential optimization for stacked EBM+XGBoost models

This script:
1. First tests different EBM configurations and selects the best one
2. Then tests different XGBoost configurations on the residuals of the best EBM
3. Returns the best overall stacked model

The approach is designed to be efficient while finding near-optimal configurations.
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
import multiprocessing
from functools import partial
import warnings
from joblib import dump, load
from datetime import datetime

# Suppress warnings to reduce output clutter
warnings.filterwarnings('ignore')

def sample_rounds_evenly(start, end, n_samples):
    """
    Sample n_samples rounds evenly from the range [start, end)
    
    Args:
        start: First round index
        end: Last round index (exclusive)
        n_samples: Number of samples to take
        
    Returns:
        List of sampled round indices
    """
    if n_samples >= (end - start):
        # If requested samples are more than or equal to available rounds, return all rounds
        return list(range(start, end))
    
    # Calculate step size to distribute samples evenly
    step = (end - start) / n_samples
    
    # Generate sample points
    sampled_rounds = [int(start + i * step) for i in range(n_samples)]
    
    return sampled_rounds

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

def train_and_evaluate_ebm(X_train, y_train, X_test, y_test, ebm_params):
    """
    Train an EBM model and evaluate its performance.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        ebm_params: Parameters for the EBM model
        
    Returns:
        Dictionary with evaluation metrics, model and predictions
    """
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
            'model': None,
            'metrics': {
                'mae': float('nan'),
                'rmse': float('nan'),
                'r2': float('nan')
            },
            'predictions': pd.DataFrame(columns=['actual', 'predicted', 'timestamp']),
            'rows_removed': {
                'train': np.sum(~is_valid_train),
                'test': np.sum(~is_valid_test)
            }
        }
    
    try:
        # Start timing the model training
        start_time = time.time()
        
        # Train EBM model
        print("  Training EBM model...")
        ebm_model = ExplainableBoostingRegressor(**ebm_params)
        ebm_model.fit(X_train_clean, y_train_vals_clean)
        
        # Get EBM predictions on test data
        ebm_test_preds = ebm_model.predict(X_test_clean)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_vals_clean, ebm_test_preds)
        rmse = np.sqrt(mean_squared_error(y_test_vals_clean, ebm_test_preds))
        r2 = r2_score(y_test_vals_clean, ebm_test_preds)
        
        # Record training time
        training_time = time.time() - start_time
        print(f"  EBM training completed in {training_time:.2f} seconds")
        
        # Create DataFrame with predictions
        pred_df = pd.DataFrame({
            'actual': y_test_vals_clean,
            'predicted': ebm_test_preds,
            'timestamp': X_test_clean.index,
            'round': np.full(len(y_test_vals_clean), -1)  # Will be filled with round number by caller
        })
        
        # Calculate residuals for potential XGBoost training
        residuals = y_test_vals_clean - ebm_test_preds
        
        return {
            'model': ebm_model,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'training_time': training_time
            },
            'predictions': pred_df,
            'residuals': residuals,
            'X_train_clean': X_train_clean,
            'y_train_clean': y_train_vals_clean,
            'X_test_clean': X_test_clean,
            'y_test_clean': y_test_vals_clean,
            'rows_removed': {
                'train': np.sum(~is_valid_train),
                'test': np.sum(~is_valid_test)
            }
        }
    except Exception as e:
        print(f"  Error in EBM training or evaluation: {str(e)}")
        return {
            'model': None,
            'metrics': {
                'mae': float('nan'),
                'rmse': float('nan'),
                'r2': float('nan')
            },
            'predictions': pd.DataFrame(columns=['actual', 'predicted', 'timestamp']),
            'rows_removed': {
                'train': np.sum(~is_valid_train),
                'test': np.sum(~is_valid_test)
            },
            'error': str(e)
        }

def train_and_evaluate_stacked(X_train, y_train, X_test, y_test, ebm_model, xgb_params):
    """
    Train the full stacked model (EBM + XGBoost on residuals) and evaluate.
    This assumes EBM model is already trained.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        ebm_model: Pre-trained EBM model
        xgb_params: Parameters for the XGBoost model
        
    Returns:
        Dictionary with evaluation metrics, models and predictions
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Get EBM predictions on train and test data
        ebm_train_preds = ebm_model.predict(X_train)
        ebm_test_preds = ebm_model.predict(X_test)
        
        # Handle whether y_train/y_test are Series or DataFrame
        if isinstance(y_train, pd.DataFrame):
            y_train_vals = y_train.iloc[:, 0].values
        else:  # It's a Series
            y_train_vals = y_train.values
            
        if isinstance(y_test, pd.DataFrame):
            y_test_vals = y_test.iloc[:, 0].values
        else:  # It's a Series
            y_test_vals = y_test.values
        
        # Calculate residuals for training data
        train_residuals = y_train_vals - ebm_train_preds
        
        # Train XGBoost on the residuals
        print("  Training XGBoost model on residuals...")
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, train_residuals)
        
        # Get XGBoost predictions on test data (predicting residuals)
        xgb_test_preds = xgb_model.predict(X_test)
        
        # Final predictions = EBM predictions + XGBoost predictions
        final_predictions = ebm_test_preds + xgb_test_preds
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_vals, final_predictions)
        rmse = np.sqrt(mean_squared_error(y_test_vals, final_predictions))
        r2 = r2_score(y_test_vals, final_predictions)
        
        # Also calculate metrics for EBM alone for comparison
        ebm_mae = mean_absolute_error(y_test_vals, ebm_test_preds)
        ebm_rmse = np.sqrt(mean_squared_error(y_test_vals, ebm_test_preds))
        ebm_r2 = r2_score(y_test_vals, ebm_test_preds)
        
        # Calculate improvement
        mae_improvement = (ebm_mae - mae) / ebm_mae * 100 if ebm_mae > 0 else 0
        rmse_improvement = (ebm_rmse - rmse) / ebm_rmse * 100 if ebm_rmse > 0 else 0
        r2_improvement = (r2 - ebm_r2) / abs(ebm_r2) * 100 if ebm_r2 != 0 else 0
        
        # Record training time
        training_time = time.time() - start_time
        print(f"  XGBoost training completed in {training_time:.2f} seconds")
        
        # Create DataFrame with predictions
        pred_df = pd.DataFrame({
            'actual': y_test_vals,
            'predicted': final_predictions,
            'ebm_pred': ebm_test_preds,
            'xgb_pred': xgb_test_preds,
            'timestamp': X_test.index,
            'round': np.full(len(y_test_vals), -1)  # Will be filled with round number by caller
        })
        
        return {
            'ebm_model': ebm_model,
            'xgb_model': xgb_model,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'ebm_mae': ebm_mae,
                'ebm_rmse': ebm_rmse,
                'ebm_r2': ebm_r2,
                'mae_improvement': mae_improvement,
                'rmse_improvement': rmse_improvement,
                'r2_improvement': r2_improvement,
                'training_time': training_time
            },
            'predictions': pred_df
        }
    except Exception as e:
        print(f"  Error in XGBoost training or evaluation: {str(e)}")
        return {
            'ebm_model': ebm_model,
            'xgb_model': None,
            'metrics': {
                'mae': float('nan'),
                'rmse': float('nan'),
                'r2': float('nan')
            },
            'predictions': pd.DataFrame(columns=['actual', 'predicted', 'ebm_pred', 'xgb_pred', 'timestamp']),
            'error': str(e)
        }

def process_single_round_ebm(round_num, target, target_dir, x_files_dir, target_index, ebm_params):
    """
    Process a single CV round for EBM model - helper function for parallelization.
    
    Returns:
        Dictionary with round results or None if error
    """
    try:
        print(f"\nProcessing round {round_num} with EBM...")
        
        # Load data for this round
        X_train, y_train, X_test, y_test = load_cv_round(
            cv_round=round_num,
            target_dir=target_dir,
            x_files_dir=x_files_dir,
            target_index=target_index
        )

        # Train and evaluate EBM
        result = train_and_evaluate_ebm(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            ebm_params=ebm_params
        )

        # Check if we skipped this round due to not enough valid data or errors
        if result['model'] is None:
            print(f"  Skipping round {round_num} due to model training errors or insufficient valid data.")
            return None
        
        # Add round number to predictions
        result['predictions']['round'] = round_num

        # Print round results
        print(f"  Round {round_num} - MAE: {result['metrics']['mae']:.4f}, RMSE: {result['metrics']['rmse']:.4f}, R²: {result['metrics']['r2']:.4f}")

        return {
            'round_num': round_num,
            'metrics': result['metrics'],
            'predictions': result['predictions'],
            'model': result['model'],
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'clean_data': {
                'X_train_clean': result.get('X_train_clean'),
                'y_train_clean': result.get('y_train_clean'),
                'X_test_clean': result.get('X_test_clean'),
                'y_test_clean': result.get('y_test_clean'),
            }
        }
    except Exception as e:
        print(f"Error processing round {round_num}: {str(e)}")
        return None

def process_single_round_stacked(round_num, target, target_dir, x_files_dir, target_index, 
                              ebm_params, xgb_params):
    """
    Process a single CV round for full stacked model - helper function for parallelization.
    Each round gets its own independently trained EBM and XGBoost models.
    
    Returns:
        Dictionary with round results or None if error
    """
    try:
        print(f"\nProcessing round {round_num} with stacked model...")
        
        # Load data for this round
        X_train, y_train, X_test, y_test = load_cv_round(
            cv_round=round_num,
            target_dir=target_dir,
            x_files_dir=x_files_dir,
            target_index=target_index
        )

        # First train EBM model for this round
        print(f"  Training EBM model for round {round_num}...")
        ebm_model = ExplainableBoostingRegressor(**ebm_params)
        
        # Handle whether y_train is Series or DataFrame
        if isinstance(y_train, pd.DataFrame):
            y_train_vals = y_train.iloc[:, 0].values
        else:  # It's a Series
            y_train_vals = y_train.values
            
        # Fit EBM model
        ebm_model.fit(X_train, y_train_vals)
        
        # Train and evaluate stacked model
        result = train_and_evaluate_stacked(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            ebm_model=ebm_model,
            xgb_params=xgb_params
        )

        # Check if we skipped this round due to errors
        if result['xgb_model'] is None:
            print(f"  Skipping round {round_num} due to model training errors.")
            return None
        
        # Add round number to predictions
        result['predictions']['round'] = round_num

        # Print round results
        print(f"  Round {round_num} - Stacked Model: MAE: {result['metrics']['mae']:.4f}, R²: {result['metrics']['r2']:.4f}")
        print(f"  Round {round_num} - EBM Only: MAE: {result['metrics']['ebm_mae']:.4f}, R²: {result['metrics']['ebm_r2']:.4f}")
        print(f"  Round {round_num} - Improvement: MAE: {result['metrics']['mae_improvement']:.2f}%, R²: {result['metrics']['r2_improvement']:.2f}%")

        return {
            'round_num': round_num,
            'metrics': result['metrics'],
            'predictions': result['predictions'],
            'ebm_model': result['ebm_model'],
            'xgb_model': result['xgb_model']
        }
    except Exception as e:
        print(f"Error processing round {round_num} with stacked model: {str(e)}")
        return None

def evaluate_ebm_parameter_sets(target, start_round=0, end_round=None, step=1,
                              ebm_parameter_sets=None, output_dir=None,
                              organized_dir="./src/data/csv", target_index=0,
                              parallel=True, max_workers=None, sample=None):
    """
    Evaluate different EBM parameter sets and find the best one.
    
    Args:
        target: Target directory name
        start_round: First CV round to process
        end_round: Last CV round to process (None = all available)
        step: Step size for processing rounds
        ebm_parameter_sets: Dictionary of parameter sets to try for EBM
        output_dir: Directory to save results
        organized_dir: Base directory for organized files
        target_index: Index of target variable (0 or 1) for regulation up/down
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers (None = auto)
        sample: If specified, sample this many rounds evenly instead of using all rounds
        
    Returns:
        Dictionary with the best EBM parameter set and evaluation results
    """
    # Define the EBM parameter sets if not provided
    if ebm_parameter_sets is None:
        ebm_parameter_sets = {
            'ebm_base': {
                'interactions': 0,  # No interaction terms - purely additive
                'outer_bags': 8,
                'inner_bags': 0,
                'learning_rate': 0.01,
                'validation_size': 0.15,
                'min_samples_leaf': 5,
                'max_leaves': 256,
                'random_state': 42
            },
            'ebm_interactions': {
                'interactions': 15,  # Include top interaction terms
                'outer_bags': 10,
                'inner_bags': 0,
                'learning_rate': 0.01,
                'validation_size': 0.15,
                'min_samples_leaf': 3,
                'max_leaves': 256,
                'random_state': 42
            },
            'ebm_robust': {
                'interactions': 10,
                'outer_bags': 15,  # More bags for stability
                'inner_bags': 0,
                'learning_rate': 0.005,  # Slower learning rate
                'validation_size': 0.15,
                'min_samples_leaf': 10,  # More conservative
                'max_leaves': 128,  # Fewer leaves to prevent overfitting
                'random_state': 42
            },
            'ebm_fast': {
                'interactions': 0,
                'outer_bags': 4,
                'inner_bags': 0,
                'learning_rate': 0.05,
                'validation_size': 0.15,
                'min_samples_leaf': 15,
                'max_leaves': 64,
                'random_state': 42
            }
        }
    
    # Set up paths
    target_dir = os.path.join(organized_dir, target)
    x_files_dir = os.path.join(organized_dir, target)
    
    # Check if target directory exists
    if not os.path.isdir(target_dir):
        raise ValueError(f"Target directory not found: {target_dir}")
    
    # Get timestamp for this evaluation
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create output directory
    if output_dir is None:
        output_dir = f"./results/stacked_optimization/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total number of rounds
    total_rounds = get_cv_round_count(target_dir)
    
    # Set default end_round if not provided
    if end_round is None:
        end_round = total_rounds
    else:
        end_round = min(end_round, total_rounds)
    
    # Get list of rounds to process
    if sample is not None:
        rounds_to_process = sample_rounds_evenly(start_round, end_round, sample)
        print(f"Sampling {len(rounds_to_process)} rounds evenly from {start_round} to {end_round-1}")
    else:
        rounds_to_process = list(range(start_round, end_round, step))
    
    # Store results for each parameter set
    ebm_evaluation_results = {}
    
    # Try each EBM parameter set
    for param_name, ebm_params in ebm_parameter_sets.items():
        print(f"\n{'='*80}")
        print(f"Evaluating EBM parameter set: {param_name}")
        print(f"Parameters: {ebm_params}")
        print(f"{'='*80}")
        
        # Create directory for this parameter set
        param_output_dir = os.path.join(output_dir, f"{target}_ind_{target_index}_{param_name}")
        os.makedirs(param_output_dir, exist_ok=True)
        
        # Save configuration
        config = {
            'target': target,
            'target_index': target_index,
            'parameter_set': param_name,
            'ebm_params': ebm_params,
            'timestamp': timestamp,
            'rounds': rounds_to_process,
            'sampled': sample is not None
        }
        
        with open(os.path.join(param_output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Process rounds
        if parallel and len(rounds_to_process) > 1:
            # Use multiprocessing for parallelism
            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count(), len(rounds_to_process))
            
            print(f"Running with {max_workers} parallel workers")
            
            # Create a partial function with fixed arguments
            process_func = partial(
                process_single_round_ebm,
                target=target,
                target_dir=target_dir,
                x_files_dir=x_files_dir,
                target_index=target_index,
                ebm_params=ebm_params
            )
            
            # Run in parallel
            with multiprocessing.Pool(max_workers) as pool:
                results = pool.map(process_func, rounds_to_process)
            
            # Filter out None results (failed rounds)
            results = [r for r in results if r is not None]
            
        else:
            # Process sequentially
            results = []
            
            for round_num in rounds_to_process:
                result = process_single_round_ebm(
                    round_num,
                    target,
                    target_dir,
                    x_files_dir,
                    target_index,
                    ebm_params
                )
                
                if result is not None:
                    results.append(result)
        
        # Process the results
        if not results:
            print(f"No successful rounds for parameter set {param_name}")
            continue
        
        # Extract metrics
        metrics = {
            'mae': [r['metrics']['mae'] for r in results],
            'rmse': [r['metrics']['rmse'] for r in results],
            'r2': [r['metrics']['r2'] for r in results],
            'training_time': [r['metrics'].get('training_time', 0) for r in results]
        }
        
        # Calculate overall metrics
        all_predictions = pd.concat([r['predictions'] for r in results], ignore_index=True)
        
        overall_actual = all_predictions['actual'].values
        overall_predicted = all_predictions['predicted'].values
        
        overall_mae = mean_absolute_error(overall_actual, overall_predicted)
        overall_rmse = np.sqrt(mean_squared_error(overall_actual, overall_predicted))
        overall_r2 = r2_score(overall_actual, overall_predicted)
        
        # Save combined predictions
        all_predictions.to_csv(os.path.join(param_output_dir, 'all_predictions.csv'), index=False)
        
        # Save combined evaluation plot
        plt.figure(figsize=(10, 8))
        plt.scatter(overall_actual, overall_predicted, alpha=0.5)
        plt.plot([min(overall_actual), max(overall_actual)], [min(overall_actual), max(overall_actual)], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'EBM {param_name} - Overall Performance (R² = {overall_r2:.4f})')
        plt.grid(True)
        plt.savefig(os.path.join(param_output_dir, 'overall_performance.png'))
        plt.close()
        
        # Summarize results
        summary = {
            'parameter_set': param_name,
            'avg_mae': np.mean(metrics['mae']),
            'avg_rmse': np.mean(metrics['rmse']),
            'avg_r2': np.mean(metrics['r2']),
            'std_mae': np.std(metrics['mae']),
            'std_rmse': np.std(metrics['rmse']),
            'std_r2': np.std(metrics['r2']),
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'overall_r2': overall_r2,
            'avg_training_time': np.mean(metrics['training_time']),
            'successful_rounds': len(results),
            'total_rounds': len(rounds_to_process),
            'sampled': sample is not None
        }
        
        # Save summary
        with open(os.path.join(param_output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Print summary
        print(f"\nSummary for parameter set {param_name}:")
        print(f"  Overall MAE: {overall_mae:.4f}")
        print(f"  Overall RMSE: {overall_rmse:.4f}")
        print(f"  Overall R²: {overall_r2:.4f}")
        print(f"  Average training time: {summary['avg_training_time']:.2f} seconds")
        
        # Store results
        ebm_evaluation_results[param_name] = {
            'params': ebm_params,
            'summary': summary,
            'results': results
        }
    
    # Find the best parameter set based on R²
    if not ebm_evaluation_results:
        raise ValueError("No successful parameter sets found")
    
    best_param_set = None
    best_r2 = -float('inf')
    
    for param_name, eval_result in ebm_evaluation_results.items():
        r2 = eval_result['summary']['overall_r2']
        if r2 > best_r2:
            best_r2 = r2
            best_param_set = param_name
    
    # Create comparison summary
    comparison = {
        'target': target,
        'target_index': target_index,
        'timestamp': timestamp,
        'best_parameter_set': best_param_set,
        'best_r2': best_r2,
        'metrics': {},
        'sampled': sample is not None,
        'sample_size': len(rounds_to_process) if sample is not None else None
    }
    
    for param_name, eval_result in ebm_evaluation_results.items():
        comparison['metrics'][param_name] = {
            'overall_mae': eval_result['summary']['overall_mae'],
            'overall_rmse': eval_result['summary']['overall_rmse'],
            'overall_r2': eval_result['summary']['overall_r2'],
            'avg_training_time': eval_result['summary']['avg_training_time']
        }
    
    # Save comparison
    with open(os.path.join(output_dir, f"{target}_ind_{target_index}_ebm_comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Create comparison bar chart
    param_names = list(ebm_evaluation_results.keys())
    r2_values = [ebm_evaluation_results[p]['summary']['overall_r2'] for p in param_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(param_names, r2_values)
    plt.ylim(min(0, min(r2_values) - 0.05), max(1, max(r2_values) + 0.05))
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('EBM Parameter Set')
    plt.ylabel('Overall R²')
    sampling_info = f" (Sampled {len(rounds_to_process)} rounds)" if sample is not None else ""
    plt.title(f'Comparison of EBM Parameter Sets for {target}, index {target_index}{sampling_info}')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target}_ind_{target_index}_ebm_comparison.png"))
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"Best EBM parameter set: {best_param_set} with R² = {best_r2:.4f}")
    print(f"{'='*80}")
    
    # Return the best parameter set and all results
    return {
        'best_parameter_set': best_param_set,
        'best_params': ebm_evaluation_results[best_param_set]['params'],
        'best_r2': best_r2,
        'all_results': ebm_evaluation_results,
        'comparison': comparison,
        'output_dir': output_dir,
        'sampled': sample is not None,
        'sample_size': len(rounds_to_process) if sample is not None else None
    }

def evaluate_stacked_model(target, best_ebm_params, start_round=0, end_round=None, step=1,
                         xgb_parameter_sets=None, output_dir=None,
                         organized_dir="./src/data/csv", target_index=0,
                         parallel=True, max_workers=None, sample=None):
    """
    Evaluate different XGBoost parameter sets for the residuals model using the best EBM model.
    
    Args:
        target: Target directory name
        best_ebm_params: Parameters for the best EBM model
        start_round: First CV round to process
        end_round: Last CV round to process (None = all available)
        step: Step size for processing rounds
        xgb_parameter_sets: Dictionary of parameter sets to try for XGBoost
        output_dir: Directory to save results
        organized_dir: Base directory for organized files
        target_index: Index of target variable (0 or 1) for regulation up/down
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers (None = auto)
        sample: If specified, sample this many rounds evenly instead of using all rounds
        
    Returns:
        Dictionary with the best stacked model configuration and evaluation results
    """
    # Define the XGBoost parameter sets if not provided
    if xgb_parameter_sets is None:
        xgb_parameter_sets = {
            'xgb_residual_light': {
                'objective': 'reg:squarederror',
                'learning_rate': 0.03,
                'n_estimators': 300,  # Fewer estimators for residuals
                'max_depth': 3,  # Shallower trees
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'random_state': 42
            },
            'xgb_residual_huber': {
                'objective': 'reg:pseudohubererror',
                'learning_rate': 0.02,
                'n_estimators': 400,
                'max_depth': 4,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'gamma': 0.1,
                'random_state': 42
            },
            'xgb_residual_quantile': {
                'objective': 'reg:quantileerror',
                'quantile_alpha': 0.5,
                'learning_rate': 0.015,
                'n_estimators': 500,
                'max_depth': 4,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'colsample_bylevel': 0.75,
                'random_state': 42
            }
        }
    
    # Set up paths
    target_dir = os.path.join(organized_dir, target)
    x_files_dir = os.path.join(organized_dir, target)
    
    # Check if target directory exists
    if not os.path.isdir(target_dir):
        raise ValueError(f"Target directory not found: {target_dir}")
    
    # Get timestamp for this evaluation if output_dir not provided
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = f"./results/stacked_optimization/{timestamp}"
    
    # Create subdirectory for XGBoost evaluation
    xgb_output_dir = os.path.join(output_dir, 'xgboost_eval')
    os.makedirs(xgb_output_dir, exist_ok=True)
    
    # Get total number of rounds
    total_rounds = get_cv_round_count(target_dir)
    
    # Set default end_round if not provided
    if end_round is None:
        end_round = total_rounds
    else:
        end_round = min(end_round, total_rounds)
    
    # Get list of rounds to process
    if sample is not None:
        rounds_to_process = sample_rounds_evenly(start_round, end_round, sample)
        print(f"Sampling {len(rounds_to_process)} rounds evenly from {start_round} to {end_round-1}")
    else:
        rounds_to_process = list(range(start_round, end_round, step))
    
    # First, train a global EBM model using all rounds
    print(f"\n{'='*80}")
    print(f"Training global EBM model for target: {target}, index: {target_index}")
    print(f"Using parameters: {best_ebm_params}")
    print(f"{'='*80}")
    
    # Load and combine data from all rounds
    all_X_train = []
    all_y_train = []
    
    for round_num in rounds_to_process:
        try:
            X_train, y_train, _, _ = load_cv_round(
                cv_round=round_num,
                target_dir=target_dir,
                x_files_dir=x_files_dir,
                target_index=target_index
            )
            all_X_train.append(X_train)
            all_y_train.append(y_train)
        except Exception as e:
            print(f"Error loading round {round_num}: {e}")
    
    if not all_X_train:
        raise ValueError("Could not load any rounds for training global EBM model")
    
    # Combine the data
    X_train_combined = pd.concat(all_X_train)
    
    # Handle whether y_train items are Series or DataFrame
    if isinstance(all_y_train[0], pd.DataFrame):
        y_train_combined = pd.concat(all_y_train)
    else:
        y_train_combined = pd.concat([pd.DataFrame(y) for y in all_y_train])
    
    if isinstance(y_train_combined, pd.DataFrame):
        y_train_vals = y_train_combined.iloc[:, 0].values
    else:
        y_train_vals = y_train_combined.values
    
    # Train global EBM model
    global_ebm = ExplainableBoostingRegressor(**best_ebm_params)
    global_ebm.fit(X_train_combined, y_train_vals)
    
    # Save the model
    dump(global_ebm, os.path.join(xgb_output_dir, f"{target}_ind_{target_index}_global_ebm.joblib"))
    
    print(f"Global EBM model trained on {len(X_train_combined)} samples")
    
    # Store results for each XGBoost parameter set
    xgb_evaluation_results = {}
    
    # Try each XGBoost parameter set
    for param_name, xgb_params in xgb_parameter_sets.items():
        print(f"\n{'='*80}")
        print(f"Evaluating XGBoost parameter set: {param_name}")
        print(f"Parameters: {xgb_params}")
        print(f"{'='*80}")
        
        # Create directory for this parameter set
        param_output_dir = os.path.join(xgb_output_dir, f"{target}_ind_{target_index}_{param_name}")
        os.makedirs(param_output_dir, exist_ok=True)
        
        # Save configuration
        config = {
            'target': target,
            'target_index': target_index,
            'parameter_set': param_name,
            'ebm_params': best_ebm_params,
            'xgb_params': xgb_params,
            'timestamp': time.strftime("%Y%m%d-%H%M%S"),
            'rounds': rounds_to_process,
            'sampled': sample is not None,
            'sample_size': len(rounds_to_process) if sample is not None else None
        }
        
        with open(os.path.join(param_output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Process rounds
        if parallel and len(rounds_to_process) > 1:
            # Use multiprocessing for parallelism
            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count(), len(rounds_to_process))
            
            print(f"Running with {max_workers} parallel workers")
            
            # Create a partial function with fixed arguments
            process_func = partial(
                process_single_round_stacked,
                target=target,
                target_dir=target_dir,
                x_files_dir=x_files_dir,
                target_index=target_index,
                ebm_model=global_ebm,
                xgb_params=xgb_params
            )
            
            # Run in parallel
            with multiprocessing.Pool(max_workers) as pool:
                results = pool.map(process_func, rounds_to_process)
            
            # Filter out None results (failed rounds)
            results = [r for r in results if r is not None]
            
        else:
            # Process sequentially
            results = []
            
            for round_num in rounds_to_process:
                result = process_single_round_stacked(
                    round_num,
                    target,
                    target_dir,
                    x_files_dir,
                    target_index,
                    global_ebm,
                    xgb_params
                )
                
                if result is not None:
                    results.append(result)
        
        # Process the results
        if not results:
            print(f"No successful rounds for parameter set {param_name}")
            continue
        
        # Extract metrics
        metrics = {
            'mae': [r['metrics']['mae'] for r in results],
            'rmse': [r['metrics'].get('rmse', float('nan')) for r in results],
            'r2': [r['metrics']['r2'] for r in results],
            'ebm_mae': [r['metrics']['ebm_mae'] for r in results],
            'ebm_r2': [r['metrics']['ebm_r2'] for r in results],
            'mae_improvement': [r['metrics']['mae_improvement'] for r in results],
            'r2_improvement': [r['metrics']['r2_improvement'] for r in results],
            'training_time': [r['metrics'].get('training_time', 0) for r in results]
        }
        
        # Calculate overall metrics
        all_predictions = pd.concat([r['predictions'] for r in results], ignore_index=True)
        
        overall_actual = all_predictions['actual'].values
        overall_predicted = all_predictions['predicted'].values
        overall_ebm_pred = all_predictions['ebm_pred'].values
        
        overall_mae = mean_absolute_error(overall_actual, overall_predicted)
        overall_rmse = np.sqrt(mean_squared_error(overall_actual, overall_predicted))
        overall_r2 = r2_score(overall_actual, overall_predicted)
        
        overall_ebm_mae = mean_absolute_error(overall_actual, overall_ebm_pred)
        overall_ebm_rmse = np.sqrt(mean_squared_error(overall_actual, overall_ebm_pred))
        overall_ebm_r2 = r2_score(overall_actual, overall_ebm_pred)
        
        overall_mae_improvement = (overall_ebm_mae - overall_mae) / overall_ebm_mae * 100
        overall_rmse_improvement = (overall_ebm_rmse - overall_rmse) / overall_ebm_rmse * 100
        overall_r2_improvement = (overall_r2 - overall_ebm_r2) / abs(overall_ebm_r2) * 100 if overall_ebm_r2 != 0 else 0
        
        # Save combined predictions
        all_predictions.to_csv(os.path.join(param_output_dir, 'all_predictions.csv'), index=False)
        
        # Save combined evaluation plot
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Stacked Model Performance
        plt.subplot(2, 2, 1)
        plt.scatter(overall_actual, overall_predicted, alpha=0.5)
        plt.plot([min(overall_actual), max(overall_actual)], [min(overall_actual), max(overall_actual)], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Stacked Model Performance (R² = {overall_r2:.4f})')
        plt.grid(True)
        
        # Plot 2: EBM Only Performance
        plt.subplot(2, 2, 2)
        plt.scatter(overall_actual, overall_ebm_pred, alpha=0.5)
        plt.plot([min(overall_actual), max(overall_actual)], [min(overall_actual), max(overall_actual)], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'EBM Only Performance (R² = {overall_ebm_r2:.4f})')
        plt.grid(True)
        
        # Plot 3: Improvement in R² Across Rounds
        plt.subplot(2, 2, 3)
        r2_improvements = [r['metrics']['r2_improvement'] for r in results]
        round_nums = [r['round_num'] for r in results]
        plt.bar(round_nums, r2_improvements)
        plt.axhline(y=overall_r2_improvement, color='r', linestyle='-', label=f'Avg: {overall_r2_improvement:.2f}%')
        plt.xlabel('Round Number')
        plt.ylabel('R² Improvement (%)')
        plt.title('R² Improvement by Round')
        plt.legend()
        plt.grid(True, axis='y')
        
        # Plot 4: Improvement in MAE Across Rounds
        plt.subplot(2, 2, 4)
        mae_improvements = [r['metrics']['mae_improvement'] for r in results]
        plt.bar(round_nums, mae_improvements)
        plt.axhline(y=overall_mae_improvement, color='r', linestyle='-', label=f'Avg: {overall_mae_improvement:.2f}%')
        plt.xlabel('Round Number')
        plt.ylabel('MAE Improvement (%)')
        plt.title('MAE Improvement by Round')
        plt.legend()
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(param_output_dir, 'stacked_model_performance.png'))
        plt.close()
        
        # Save the model
        dump(
            {
                'ebm_model': global_ebm,
                'xgb_model': None,  # We don't save individual round models
                'ebm_params': best_ebm_params,
                'xgb_params': xgb_params
            },
            os.path.join(param_output_dir, f"{target}_ind_{target_index}_stacked_model.joblib")
        )
        
        # Summarize results
        summary = {
            'ebm_parameter_set': 'best_ebm',
            'xgb_parameter_set': param_name,
            'avg_mae': np.mean(metrics['mae']),
            'avg_rmse': np.mean(metrics['rmse']),
            'avg_r2': np.mean(metrics['r2']),
            'avg_ebm_mae': np.mean(metrics['ebm_mae']),
            'avg_ebm_r2': np.mean(metrics['ebm_r2']),
            'avg_mae_improvement': np.mean(metrics['mae_improvement']),
            'avg_r2_improvement': np.mean(metrics['r2_improvement']),
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'overall_r2': overall_r2,
            'overall_ebm_mae': overall_ebm_mae,
            'overall_ebm_rmse': overall_ebm_rmse,
            'overall_ebm_r2': overall_ebm_r2,
            'overall_mae_improvement': overall_mae_improvement,
            'overall_rmse_improvement': overall_rmse_improvement,
            'overall_r2_improvement': overall_r2_improvement,
            'avg_training_time': np.mean(metrics['training_time']),
            'successful_rounds': len(results),
            'total_rounds': len(rounds_to_process),
            'sampled': sample is not None,
            'sample_size': len(rounds_to_process) if sample is not None else None
        }
        
        # Save summary
        with open(os.path.join(param_output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Print summary
        print(f"\nSummary for XGBoost parameter set {param_name}:")
        print(f"  Overall Stacked Model MAE: {overall_mae:.4f}")
        print(f"  Overall Stacked Model R²: {overall_r2:.4f}")
        print(f"  Overall EBM Only R²: {overall_ebm_r2:.4f}")
        print(f"  Overall R² Improvement: {overall_r2_improvement:.2f}%")
        print(f"  Overall MAE Improvement: {overall_mae_improvement:.2f}%")
        
        # Store results
        xgb_evaluation_results[param_name] = {
            'ebm_params': best_ebm_params,
            'xgb_params': xgb_params,
            'summary': summary
        }
    
    # Find the best parameter set based on R² improvement
    if not xgb_evaluation_results:
        raise ValueError("No successful XGBoost parameter sets found")
    
    best_param_set = None
    best_r2_improvement = -float('inf')
    
    for param_name, eval_result in xgb_evaluation_results.items():
        r2_improvement = eval_result['summary']['overall_r2_improvement']
        if r2_improvement > best_r2_improvement:
            best_r2_improvement = r2_improvement
            best_param_set = param_name
    
    # Create comparison summary
    comparison = {
        'target': target,
        'target_index': target_index,
        'timestamp': time.strftime("%Y%m%d-%H%M%S"),
        'best_xgb_parameter_set': best_param_set,
        'best_r2_improvement': best_r2_improvement,
        'metrics': {},
        'sampled': sample is not None,
        'sample_size': len(rounds_to_process) if sample is not None else None
    }
    
    for param_name, eval_result in xgb_evaluation_results.items():
        comparison['metrics'][param_name] = {
            'overall_mae': eval_result['summary']['overall_mae'],
            'overall_r2': eval_result['summary']['overall_r2'],
            'overall_ebm_r2': eval_result['summary']['overall_ebm_r2'],
            'overall_r2_improvement': eval_result['summary']['overall_r2_improvement'],
            'overall_mae_improvement': eval_result['summary']['overall_mae_improvement']
        }
    
    # Save comparison
    with open(os.path.join(xgb_output_dir, f"{target}_ind_{target_index}_xgb_comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Create comparison bar chart
    param_names = list(xgb_evaluation_results.keys())
    r2_improvements = [xgb_evaluation_results[p]['summary']['overall_r2_improvement'] for p in param_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(param_names, r2_improvements)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('XGBoost Parameter Set')
    plt.ylabel('R² Improvement (%)')
    sampling_info = f" (Sampled {len(rounds_to_process)} rounds)" if sample is not None else ""
    plt.title(f'Comparison of XGBoost Parameter Sets for {target}, index {target_index}{sampling_info}')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(xgb_output_dir, f"{target}_ind_{target_index}_xgb_comparison.png"))
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"Best XGBoost parameter set: {best_param_set} with R² improvement = {best_r2_improvement:.2f}%")
    print(f"{'='*80}")
    
    # Save the best stacked model
    best_stacked_model = {
        'ebm_model': global_ebm,
        'ebm_params': best_ebm_params,
        'xgb_params': xgb_evaluation_results[best_param_set]['xgb_params'],
        'target': target,
        'target_index': target_index
    }
    
    dump(best_stacked_model, os.path.join(xgb_output_dir, f"{target}_ind_{target_index}_best_stacked_model.joblib"))
    
    # Final overall summary
    final_summary = {
        'target': target,
        'target_index': target_index,
        'best_ebm_params': best_ebm_params,
        'best_xgb_params': xgb_evaluation_results[best_param_set]['xgb_params'],
        'ebm_only_r2': xgb_evaluation_results[best_param_set]['summary']['overall_ebm_r2'],
        'stacked_r2': xgb_evaluation_results[best_param_set]['summary']['overall_r2'],
        'r2_improvement': best_r2_improvement,
        'timestamp': time.strftime("%Y%m%d-%H%M%S"),
        'sampled': sample is not None,
        'sample_size': len(rounds_to_process) if sample is not None else None
    }
    
    # Save final summary
    with open(os.path.join(output_dir, f"{target}_ind_{target_index}_final_summary.json"), 'w') as f:
        json.dump(final_summary, f, indent=4)
    
    # Create final comparison plot
    plt.figure(figsize=(8, 6))
    models = ['EBM Only', 'Stacked Model']
    r2_values = [xgb_evaluation_results[best_param_set]['summary']['overall_ebm_r2'],
                xgb_evaluation_results[best_param_set]['summary']['overall_r2']]
    
    bars = plt.bar(models, r2_values, color=['blue', 'green'])
    plt.ylim(min(0, min(r2_values) - 0.05), max(1, max(r2_values) + 0.05))
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.ylabel('R²')
    sampling_info = f" (Sampled {len(rounds_to_process)} rounds)" if sample is not None else ""
    plt.title(f'Performance Comparison for {target}, index {target_index}{sampling_info}')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Add improvement text
    plt.figtext(0.5, 0.01, f"Improvement: {best_r2_improvement:.2f}%", 
               ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2})
    
    plt.grid(axis='y')
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the improvement text
    plt.savefig(os.path.join(output_dir, f"{target}_ind_{target_index}_final_comparison.png"))
    plt.close()
    
    return {
        'best_ebm_params': best_ebm_params,
        'best_xgb_params': xgb_evaluation_results[best_param_set]['xgb_params'],
        'ebm_only_r2': xgb_evaluation_results[best_param_set]['summary']['overall_ebm_r2'],
        'stacked_r2': xgb_evaluation_results[best_param_set]['summary']['overall_r2'],
        'r2_improvement': best_r2_improvement,
        'ebm_results': xgb_evaluation_results[best_param_set]['summary'],
        'output_dir': output_dir,
        'sampled': sample is not None,
        'sample_size': len(rounds_to_process) if sample is not None else None
    }

def optimize_stacked_model(target, start_round=0, end_round=None, step=1,
                         ebm_parameter_sets=None, xgb_parameter_sets=None,
                         organized_dir="./src/data/csv", target_index=0,
                         parallel=True, max_workers=None, sample=None):
    """
    Optimize a stacked EBM+XGBoost model for a target.
    First finds the best EBM model, then the best XGBoost model for the residuals.
    
    Args:
        target: Target directory name
        start_round: First CV round to process
        end_round: Last CV round to process (None = all available)
        step: Step size for processing rounds
        ebm_parameter_sets: Dictionary of parameter sets to try for EBM
        xgb_parameter_sets: Dictionary of parameter sets to try for XGBoost
        organized_dir: Base directory for organized files
        target_index: Index of target variable (0 or 1) for regulation up/down
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers (None = auto)
        sample: If specified, sample this many rounds evenly instead of using all rounds
        
    Returns:
        Dictionary with the optimized model configuration and evaluation results
    """
    # Create timestamp for this optimization run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"./results/stacked_optimization/{timestamp}"
    
    # Step 1: Find the best EBM model
    print(f"\n{'='*80}")
    print(f"Step 1: Finding the best EBM model for target: {target}, index: {target_index}")
    sampling_info = f" (Sampling {sample} rounds)" if sample is not None else ""
    print(f"Rounds: {start_round} to {end_round if end_round is not None else 'all'}{sampling_info}")
    print(f"{'='*80}")
    
    ebm_results = evaluate_ebm_parameter_sets(
        target=target,
        start_round=start_round,
        end_round=end_round,
        step=step,
        ebm_parameter_sets=ebm_parameter_sets,
        output_dir=output_dir,
        organized_dir=organized_dir,
        target_index=target_index,
        parallel=parallel,
        max_workers=max_workers,
        sample=sample
    )
    
    # Step 2: Find the best XGBoost model for the residuals
    print(f"\n{'='*80}")
    print(f"Step 2: Finding the best XGBoost model for the residuals")
    print(f"Using best EBM model: {ebm_results['best_parameter_set']}")
    print(f"{'='*80}")
    
    stacked_results = evaluate_stacked_model(
        target=target,
        best_ebm_params=ebm_results['best_params'],
        start_round=start_round,
        end_round=end_round,
        step=step,
        xgb_parameter_sets=xgb_parameter_sets,
        output_dir=output_dir,
        organized_dir=organized_dir,
        target_index=target_index,
        parallel=parallel,
        max_workers=max_workers,
        sample=sample
    )
    
    # Combine results
    optimization_results = {
        'target': target,
        'target_index': target_index,
        'best_ebm_parameter_set': ebm_results['best_parameter_set'],
        'best_ebm_params': ebm_results['best_params'],
        'best_xgb_params': stacked_results['best_xgb_params'],
        'ebm_only_r2': stacked_results['ebm_only_r2'],
        'stacked_r2': stacked_results['stacked_r2'],
        'r2_improvement': stacked_results['r2_improvement'],
        'output_dir': output_dir,
        'timestamp': timestamp,
        'sampled': sample is not None,
        'sample_size': stacked_results.get('sample_size')
    }
    
    # Save the final optimization results
    with open(os.path.join(output_dir, f"{target}_ind_{target_index}_optimization_results.json"), 'w') as f:
        json.dump(optimization_results, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"Optimization complete for target: {target}, index: {target_index}")
    print(f"Best EBM model: {ebm_results['best_parameter_set']}")
    print(f"EBM Only R²: {stacked_results['ebm_only_r2']:.4f}")
    print(f"Stacked Model R²: {stacked_results['stacked_r2']:.4f}")
    print(f"R² Improvement: {stacked_results['r2_improvement']:.2f}%")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")
    
    return optimization_results

def run_stacked_model_with_params(target, ebm_params, xgb_params, start_round=0, end_round=None, step=1,
                                output_dir=None, organized_dir="./src/data/csv", target_index=0,
                                parallel=True, max_workers=None, sample=None):
    """
    Run the stacked model with specified parameters on all rounds.
    Each round gets its own independently trained models.
    
    Args:
        target: Target directory name
        ebm_params: Parameters for the EBM model
        xgb_params: Parameters for the XGBoost model
        start_round: First CV round to process
        end_round: Last CV round to process (None = all available)
        step: Step size for processing rounds
        output_dir: Directory to save results
        organized_dir: Base directory for organized files
        target_index: Index of target variable (0 or 1) for regulation up/down
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers (None = auto)
        sample: If specified, sample this many rounds evenly instead of using all rounds
        
    Returns:
        Dictionary with results summary
    """
    # Set up paths
    target_dir = os.path.join(organized_dir, target)
    x_files_dir = os.path.join(organized_dir, target)
    
    # Check if target directory exists
    if not os.path.isdir(target_dir):
        raise ValueError(f"Target directory not found: {target_dir}")
    
    # Get timestamp for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create output directory
    if output_dir is None:
        output_dir = f"./results/stacked/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectory for this target and index
    target_output_dir = os.path.join(output_dir, f"{target}_ind_{target_index}")
    os.makedirs(target_output_dir, exist_ok=True)
    
    # Get total number of rounds
    total_rounds = get_cv_round_count(target_dir)
    
    # Set default end_round if not provided
    if end_round is None:
        end_round = total_rounds
    else:
        end_round = min(end_round, total_rounds)
    
    # Get list of rounds to process
    if sample is not None:
        rounds_to_process = sample_rounds_evenly(start_round, end_round, sample)
        print(f"Sampling {len(rounds_to_process)} rounds evenly from {start_round} to {end_round-1}")
    else:
        rounds_to_process = list(range(start_round, end_round, step))
    
    # Save configuration
    config = {
        'target': target,
        'target_index': target_index,
        'ebm_params': ebm_params,
        'xgb_params': xgb_params,
        'timestamp': timestamp,
        'start_round': start_round,
        'end_round': end_round,
        'step': step,
        'rounds_processed': rounds_to_process,
        'sampled': sample is not None,
        'sample_size': len(rounds_to_process) if sample is not None else None
    }
    
    with open(os.path.join(target_output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"Running stacked model for target: {target}, index: {target_index}")
    sampling_info = f" (Sampled {len(rounds_to_process)} rounds)" if sample is not None else ""
    print(f"Rounds: {start_round} to {end_round-1} with step {step}{sampling_info}")
    print(f"Total rounds to process: {len(rounds_to_process)}")
    print(f"{'='*80}")
    
    # Process each round
    if parallel and len(rounds_to_process) > 1:
        # Use multiprocessing for parallelism
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(rounds_to_process))
        
        print(f"Running with {max_workers} parallel workers")
        
        # Create a partial function with fixed arguments
        process_func = partial(
            process_single_round_stacked,
            target=target,
            target_dir=target_dir,
            x_files_dir=x_files_dir,
            target_index=target_index,
            ebm_params=ebm_params,
            xgb_params=xgb_params
        )
        
        # Run in parallel
        with multiprocessing.Pool(max_workers) as pool:
            results = pool.map(process_func, rounds_to_process)
        
        # Filter out None results (failed rounds)
        results = [r for r in results if r is not None]
        
    else:
        # Process sequentially
        results = []
        
        for round_num in rounds_to_process:
            result = process_single_round_stacked(
                round_num,
                target,
                target_dir,
                x_files_dir,
                target_index,
                ebm_params,
                xgb_params
            )
            
            if result is not None:
                results.append(result)
    
    # Check if we have any successful rounds
    if not results:
        print(f"No successful rounds for target {target}, index {target_index}")
        return {
            'error': 'No successful rounds',
            'target': target,
            'target_index': target_index
        }
    
    # Save individual round results
    for result in results:
        round_num = result['round_num']
        
        # Save predictions
        result['predictions'].to_csv(os.path.join(target_output_dir, f"round_{round_num}_predictions.csv"), index=False)
        
        # Create and save plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(result['predictions']['timestamp'], result['predictions']['actual'], 'b-', label='Actual')
        plt.plot(result['predictions']['timestamp'], result['predictions']['predicted'], 'r--', label='Stacked Model')
        plt.plot(result['predictions']['timestamp'], result['predictions']['ebm_pred'], 'g-.', label='EBM Only')
        plt.title(f'Round {round_num} - Actual vs Predictions')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.scatter(result['predictions']['actual'], result['predictions']['predicted'], label='Stacked')
        plt.scatter(result['predictions']['actual'], result['predictions']['ebm_pred'], alpha=0.5, label='EBM Only')
        plt.plot([min(result['predictions']['actual']), max(result['predictions']['actual'])], 
                [min(result['predictions']['actual']), max(result['predictions']['actual'])], 'k--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Round {round_num} - Correlation')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, f"round_{round_num}_plot.png"))
        plt.close()
    
    # Calculate overall metrics
    all_predictions = pd.concat([r['predictions'] for r in results], ignore_index=True)
    
    overall_actual = all_predictions['actual'].values
    overall_predicted = all_predictions['predicted'].values
    overall_ebm_pred = all_predictions['ebm_pred'].values
    
    overall_mae = mean_absolute_error(overall_actual, overall_predicted)
    overall_rmse = np.sqrt(mean_squared_error(overall_actual, overall_predicted))
    overall_r2 = r2_score(overall_actual, overall_predicted)
    
    overall_ebm_mae = mean_absolute_error(overall_actual, overall_ebm_pred)
    overall_ebm_rmse = np.sqrt(mean_squared_error(overall_actual, overall_ebm_pred))
    overall_ebm_r2 = r2_score(overall_actual, overall_ebm_pred)
    
    overall_mae_improvement = (overall_ebm_mae - overall_mae) / overall_ebm_mae * 100
    overall_rmse_improvement = (overall_ebm_rmse - overall_rmse) / overall_ebm_rmse * 100
    overall_r2_improvement = (overall_r2 - overall_ebm_r2) / abs(overall_ebm_r2) * 100 if overall_ebm_r2 != 0 else 0
    
    # Save combined predictions
    all_predictions.to_csv(os.path.join(target_output_dir, 'all_predictions.csv'), index=False)
    
    # Create overall evaluation plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Stacked Model Performance
    plt.subplot(2, 2, 1)
    plt.scatter(overall_actual, overall_predicted, alpha=0.5)
    plt.plot([min(overall_actual), max(overall_actual)], [min(overall_actual), max(overall_actual)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    sampling_info = f" (Sampled {len(rounds_to_process)} rounds)" if sample is not None else ""
    plt.title(f'Stacked Model Performance (R² = {overall_r2:.4f}){sampling_info}')
    plt.grid(True)
    
    # Plot 2: EBM Only Performance
    plt.subplot(2, 2, 2)
    plt.scatter(overall_actual, overall_ebm_pred, alpha=0.5)
    plt.plot([min(overall_actual), max(overall_actual)], [min(overall_actual), max(overall_actual)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'EBM Only Performance (R² = {overall_ebm_r2:.4f})')
    plt.grid(True)
    
    # Plot 3: Improvement in R² Across Rounds
    plt.subplot(2, 2, 3)
    r2_improvements = [r['metrics']['r2_improvement'] for r in results]
    round_nums = [r['round_num'] for r in results]
    plt.bar(round_nums, r2_improvements)
    plt.axhline(y=overall_r2_improvement, color='r', linestyle='-', label=f'Avg: {overall_r2_improvement:.2f}%')
    plt.xlabel('Round Number')
    plt.ylabel('R² Improvement (%)')
    plt.title('R² Improvement by Round')
    plt.legend()
    plt.grid(True, axis='y')
    
    # Plot 4: Model Comparison
    plt.subplot(2, 2, 4)
    models = ['EBM Only', 'Stacked Model']
    r2_values = [overall_ebm_r2, overall_r2]
    bars = plt.bar(models, r2_values, color=['blue', 'green'])
    plt.ylim(min(0, min(r2_values) - 0.05), max(1, max(r2_values) + 0.05))
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.ylabel('R²')
    plt.title('Performance Comparison')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(target_output_dir, 'overall_performance.png'))
    plt.close()
    
    # Extract metrics from results
    metrics = {
        'mae': [r['metrics']['mae'] for r in results],
        'rmse': [r['metrics'].get('rmse', float('nan')) for r in results],
        'r2': [r['metrics']['r2'] for r in results],
        'ebm_mae': [r['metrics']['ebm_mae'] for r in results],
        'ebm_r2': [r['metrics']['ebm_r2'] for r in results],
        'mae_improvement': [r['metrics']['mae_improvement'] for r in results],
        'r2_improvement': [r['metrics']['r2_improvement'] for r in results]
    }
    
    # Create metrics summary plot
    plt.figure(figsize=(12, 10))
    
    # Plot 1: MAE by Round
    plt.subplot(2, 2, 1)
    plt.bar(round_nums, metrics['mae'], color='blue', alpha=0.7, label='Stacked Model')
    plt.bar(round_nums, metrics['ebm_mae'], color='green', alpha=0.4, label='EBM Only')
    plt.axhline(y=overall_mae, color='blue', linestyle='-', label=f'Stacked Avg: {overall_mae:.4f}')
    plt.axhline(y=overall_ebm_mae, color='green', linestyle='-', label=f'EBM Avg: {overall_ebm_mae:.4f}')
    plt.xlabel('Round Number')
    plt.ylabel('MAE')
    plt.title('MAE by Round')
    plt.legend()
    plt.grid(True, axis='y')
    
    # Plot 2: R² by Round
    plt.subplot(2, 2, 2)
    plt.bar(round_nums, metrics['r2'], color='blue', alpha=0.7, label='Stacked Model')
    plt.bar(round_nums, metrics['ebm_r2'], color='green', alpha=0.4, label='EBM Only')
    plt.axhline(y=overall_r2, color='blue', linestyle='-', label=f'Stacked Avg: {overall_r2:.4f}')
    plt.axhline(y=overall_ebm_r2, color='green', linestyle='-', label=f'EBM Avg: {overall_ebm_r2:.4f}')
    plt.xlabel('Round Number')
    plt.ylabel('R²')
    plt.title('R² by Round')
    plt.legend()
    plt.grid(True, axis='y')
    
    # Plot 3: MAE Improvement by Round
    plt.subplot(2, 2, 3)
    plt.bar(round_nums, metrics['mae_improvement'])
    plt.axhline(y=overall_mae_improvement, color='r', linestyle='-', label=f'Avg: {overall_mae_improvement:.2f}%')
    plt.xlabel('Round Number')
    plt.ylabel('MAE Improvement (%)')
    plt.title('MAE Improvement by Round')
    plt.legend()
    plt.grid(True, axis='y')
    
    # Plot 4: R² Improvement by Round
    plt.subplot(2, 2, 4)
    plt.bar(round_nums, metrics['r2_improvement'])
    plt.axhline(y=overall_r2_improvement, color='r', linestyle='-', label=f'Avg: {overall_r2_improvement:.2f}%')
    plt.xlabel('Round Number')
    plt.ylabel('R² Improvement (%)')
    plt.title('R² Improvement by Round')
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(target_output_dir, 'metrics_summary.png'))
    plt.close()
    
    # Create summary
    summary = {
        'target': target,
        'target_index': target_index,
        'ebm_params': ebm_params,
        'xgb_params': xgb_params,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'overall_r2': overall_r2,
        'overall_ebm_mae': overall_ebm_mae,
        'overall_ebm_rmse': overall_ebm_rmse,
        'overall_ebm_r2': overall_ebm_r2,
        'overall_mae_improvement': overall_mae_improvement,
        'overall_rmse_improvement': overall_rmse_improvement,
        'overall_r2_improvement': overall_r2_improvement,
        'avg_mae': np.mean(metrics['mae']),
        'avg_rmse': np.mean([m for m in metrics['rmse'] if not np.isnan(m)]),
        'avg_r2': np.mean(metrics['r2']),
        'avg_ebm_mae': np.mean(metrics['ebm_mae']),
        'avg_ebm_r2': np.mean(metrics['ebm_r2']),
        'avg_mae_improvement': np.mean(metrics['mae_improvement']),
        'avg_r2_improvement': np.mean(metrics['r2_improvement']),
        'std_mae': np.std(metrics['mae']),
        'std_r2': np.std(metrics['r2']),
        'successful_rounds': len(results),
        'total_rounds_attempted': len(rounds_to_process),
        'rounds_processed': [r['round_num'] for r in results],
        'timestamp': timestamp,
        'sampled': sample is not None,
        'sample_size': len(rounds_to_process) if sample is not None else None
    }
    
    # Save summary
    with open(os.path.join(target_output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save model parameters (not the actual models)
    model_params = {
        'ebm_params': ebm_params,
        'xgb_params': xgb_params,
        'target': target,
        'target_index': target_index
    }
    
    with open(os.path.join(target_output_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=4)
    
    # Print summary
    print("\nStacked Model Results Summary:")
    print(f"Target: {target}, Index: {target_index}")
    sampling_info = f" (Sampled {len(rounds_to_process)} rounds)" if sample is not None else ""
    print(f"Processed {summary['successful_rounds']} of {summary['total_rounds_attempted']} rounds successfully{sampling_info}")
    print(f"Overall Stacked Model R²: {overall_r2:.4f}")
    print(f"Overall EBM Only R²: {overall_ebm_r2:.4f}")
    print(f"Overall R² Improvement: {overall_r2_improvement:.2f}%")
    print(f"Overall Stacked Model MAE: {overall_mae:.4f}")
    print(f"Overall EBM Only MAE: {overall_ebm_mae:.4f}")
    print(f"Overall MAE Improvement: {overall_mae_improvement:.2f}%")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Optimize and run stacked EBM+XGBoost models')
    
    parser.add_argument('--targets', type=str, default=None,
                       help='Target to process, comma-separated (default: show available targets)')
    
    parser.add_argument('--start', type=int, default=0,
                       help='First CV round to process (default: 0)')
    
    parser.add_argument('--end', type=int, default=None,
                       help='Last CV round to process (default: all available)')
    
    parser.add_argument('--step', type=int, default=1,
                       help='Step size for processing rounds (default: 1)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Base directory to save results')
    
    parser.add_argument('--organized-dir', type=str, default='./src/data/csv',
                       help='Base directory for organized files (default: ./src/data/csv)')
    
    parser.add_argument('--list', action='store_true',
                       help='List available targets and exit')
    
    parser.add_argument('--target-index', type=int, default=None,
                        help='Index of target to process (0 or 1, default: run both)')
    
    parser.add_argument('--optimize', action='store_true',
                        help='Run sequential optimization to find the best stacked model')
    
    parser.add_argument('--run-best', action='store_true',
                        help='Run with the best previously found parameters (read from JSON file)')
    
    parser.add_argument('--best-params-file', type=str, default=None,
                        help='JSON file with best parameters (required for --run-best)')
    
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: auto)')
    
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample this many rounds evenly (default: use all rounds)')
    
    args = parser.parse_args()
    
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
    
    # Define EBM parameter sets
    ebm_parameter_sets = {
            'ebm_base': {
                'interactions': 0,  # No interaction terms - purely additive
                'outer_bags': 8,
                'inner_bags': 0,
                'learning_rate': 0.01,
                'validation_size': 0.15,
                'min_samples_leaf': 5,
                'max_leaves': 256,
                'random_state': 42
            },
            'ebm_light_interactions': {
                'interactions': 3,  # Limited interactions for reasonable speed
                'outer_bags': 8,
                'inner_bags': 0,
                'learning_rate': 0.01,
                'validation_size': 0.15,
                'min_samples_leaf': 5,
                'max_leaves': 256,
                'random_state': 42
            },
            'ebm_robust': {
                'interactions': 0,  # Removed interactions for speed
                'outer_bags': 15,  # More bags for stability
                'inner_bags': 0,
                'learning_rate': 0.005,  # Slower learning rate
                'validation_size': 0.15,
                'min_samples_leaf': 10,  # More conservative
                'max_leaves': 128,  # Fewer leaves to prevent overfitting
                'random_state': 42
            }
        }
    
    # Define XGBoost parameter sets for residual modeling
    xgb_residual_sets = {
        'xgb_residual_light': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.03,
            'n_estimators': 300,  # Fewer estimators for residuals
            'max_depth': 3,  # Shallower trees
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'random_state': 42
        },
        'xgb_residual_huber': {
            'objective': 'reg:pseudohubererror',
            'learning_rate': 0.02,
            'n_estimators': 400,
            'max_depth': 4,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'gamma': 0.1,
            'random_state': 42
        },
        'xgb_residual_quantile': {
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.5,
            'learning_rate': 0.015,
            'n_estimators': 500,
            'max_depth': 4,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'colsample_bylevel': 0.75,
            'random_state': 42
        }
    }
    
    # Determine targets to process
    target_list = available_targets if args.targets is None else args.targets.split(',')
    
    # Process each target
    for target in target_list:
        if target not in available_targets:
            print(f"Target '{target}' not found in available targets.")
            continue
        
        try:
            print(f"\n{'='*80}")
            print(f"Processing target: {target}")
            print(f"{'='*80}")
            
            # Determine target indices to process
            if args.target_index is not None:
                target_indices = [args.target_index]
            else:
                target_indices = [0, 1]  # Process both indices
            
            for target_index in target_indices:
                if args.optimize:
                    # Run sequential optimization
                    optimize_stacked_model(
                        target=target,
                        start_round=args.start,
                        end_round=args.end,
                        step=args.step,
                        ebm_parameter_sets=ebm_parameter_sets,
                        xgb_parameter_sets=xgb_residual_sets,
                        organized_dir=args.organized_dir,
                        target_index=target_index,
                        parallel=not args.no_parallel,
                        max_workers=args.max_workers,
                        sample=args.sample
                    )
                elif args.run_best:
                    # Run with best previously found parameters
                    if args.best_params_file is None:
                        print("Error: --best-params-file is required when using --run-best")
                        return
                    
                    # Load best parameters from file
                    with open(args.best_params_file, 'r') as f:
                        best_params = json.load(f)
                    
                    # Check if parameters for this target and index exist
                    target_key = f"{target}_ind_{target_index}"
                    if target_key not in best_params:
                        print(f"Error: No parameters found for {target_key} in {args.best_params_file}")
                        continue
                    
                    # Get the parameters
                    ebm_params = best_params[target_key]['ebm_params']
                    xgb_params = best_params[target_key]['xgb_params']
                    
                    print(f"Running with best parameters from {args.best_params_file}")
                    print(f"EBM params: {ebm_params}")
                    print(f"XGBoost params: {xgb_params}")
                    
                    # Run with the best parameters
                    run_stacked_model_with_params(
                        target=target,
                        ebm_params=ebm_params,
                        xgb_params=xgb_params,
                        start_round=args.start,
                        end_round=args.end,
                        step=args.step,
                        output_dir=args.output,
                        organized_dir=args.organized_dir,
                        target_index=target_index,
                        parallel=not args.no_parallel,
                        max_workers=args.max_workers,
                        sample=args.sample
                    )
                else:
                    # Run with default parameters (one set each)
                    print("Using default parameters (no optimization)")
                    
                    ebm_params = ebm_parameter_sets['ebm_base']
                    xgb_params = xgb_residual_sets['xgb_residual_light']
                    
                    run_stacked_model_with_params(
                        target=target,
                        ebm_params=ebm_params,
                        xgb_params=xgb_params,
                        start_round=args.start,
                        end_round=args.end,
                        step=args.step,
                        output_dir=args.output,
                        organized_dir=args.organized_dir,
                        target_index=target_index,
                        parallel=not args.no_parallel,
                        max_workers=args.max_workers,
                        sample=args.sample
                    )
        except Exception as e:
            print(f"Error processing target {target}: {str(e)}")

if __name__ == "__main__":
    main()