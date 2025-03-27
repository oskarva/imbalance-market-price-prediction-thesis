#!/usr/bin/env python3
"""
ebm_xgboost_cv_all_targets.py - Apply a stacked EBM+XGBoost model to multiple target areas

This script applies a stacked model to datasets across different areas:
1. An Explainable Boosting Machine (EBM) is trained as the base model
2. XGBoost is trained on the residuals of the EBM predictions
3. Final predictions are the sum of both models' outputs

The script follows the same structure as run_cv_all_targets.py but with this custom stacked model.
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

def train_and_evaluate_stacked_model(X_train, y_train, X_test, y_test, ebm_params=None, xgb_params=None):
    """
    Train and evaluate a stacked model (EBM + XGBoost on residuals) with proper data cleaning.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        ebm_params: Parameters for the EBM model
        xgb_params: Parameters for the XGBoost model
        
    Returns:
        Dictionary with evaluation metrics, models and predictions
    """
    # Default parameters if none provided
    if ebm_params is None:
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
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'random_state': 42
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
        xgb_model.fit(X_train_clean, train_residuals)
        
        # Get XGBoost predictions on test data (predicting residuals)
        xgb_test_preds = xgb_model.predict(X_test_clean)
        
        # Step 4: Final predictions = EBM predictions + XGBoost predictions
        final_predictions = ebm_test_preds + xgb_test_preds
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_vals_clean, final_predictions)
        rmse = np.sqrt(mean_squared_error(y_test_vals_clean, final_predictions))
        
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
                'rmse': rmse
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

def run_cv_for_target_stacked(target, start_round=0, end_round=None, step=1,
                            ebm_params=None, xgb_params=None, output_dir=None,
                            organized_dir="./src/data/csv", target_index=0):
    """
    Run cross-validation for a specific target with stacked model (EBM+XGBoost) and overall R² calculation.
    
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
        index_output_dir = f"./results/stacked/{target}_run_{timestamp}_ind_{target_index}"
    else:
        # If output_dir is provided, append the index
        index_output_dir = f"{output_dir}/{target}_ind_{target_index}"
    
    # Create output directory
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
        'target_index': target_index
    }
    
    with open(os.path.join(index_output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nRunning stacked model (EBM+XGBoost) cross-validation for target: {target}, index: {target_index}")
    print(f"Rounds: {start_round} to {end_round-1} with step {step}")
    print(f"Total rounds available: {total_rounds}")
    if ebm_params:
        print(f"EBM parameters: {ebm_params}")
    if xgb_params:
        print(f"XGBoost parameters: {xgb_params}")

    # Track metrics across all rounds
    all_metrics = {
        'mae': [],
        'rmse': []
    }

    # Track all predictions for overall R² calculation
    all_predictions = []

    # Process each round
    round_results = {}
    successful_rounds = 0
    failed_rounds = 0

    for round_num in range(start_round, end_round, step):
        print(f"\nProcessing round {round_num}...")

        try:
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
                xgb_params=xgb_params
            )

            # Check if we skipped this round due to not enough valid data or errors
            if result['ebm_model'] is None or result['xgb_model'] is None:
                print(f"  Skipping round {round_num} due to model training errors or insufficient valid data.")
                failed_rounds += 1
                continue
            
            # Add round number to predictions
            result['predictions']['round'] = round_num

            # Store results
            round_results[round_num] = result['metrics']

            # Update tracking metrics
            all_metrics['mae'].append(result['metrics']['mae'])
            all_metrics['rmse'].append(result['metrics']['rmse'])

            # Add to all predictions for overall R² calculation
            all_predictions.append(result['predictions'])

            # Print round results
            print(f"  MAE: {result['metrics']['mae']:.4f}")
            print(f"  RMSE: {result['metrics']['rmse']:.4f}")

            if 'rows_removed' in result:
                print(f"  Rows removed: {result['rows_removed']['train']} train, {result['rows_removed']['test']} test")

            # Save predictions
            pred_df = result['predictions']
            pred_df.to_csv(os.path.join(index_output_dir, f"round_{round_num}_predictions.csv"), index=False)

            # Create and save plot
            plt.figure(figsize=(15, 10))

            # Plot 1: Actual vs All Predictions
            plt.subplot(2, 2, 1)
            plt.plot(pred_df['timestamp'], pred_df['actual'], 'b-', label='Actual')
            plt.plot(pred_df['timestamp'], pred_df['predicted'], 'r--', label='Final Prediction')
            plt.plot(pred_df['timestamp'], pred_df['ebm_pred'], 'g-.', label='EBM Prediction')
            plt.plot(pred_df['timestamp'], pred_df['xgb_pred'], 'm:', label='XGB Residual')
            plt.title(f'Round {round_num} - Actual vs Predictions')
            plt.ylabel('Price')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)

            # Plot 2: Scatter plot to visualize final prediction correlation
            plt.subplot(2, 2, 2)
            plt.scatter(pred_df['actual'], pred_df['predicted'])
            plt.xlabel('Actual')
            plt.ylabel('Final Prediction')

            # Calculate round-specific R² for the plot title only
            if np.var(pred_df['actual']) > 0:
                r2 = r2_score(pred_df['actual'], pred_df['predicted'])
                plt.title(f'Round {round_num} Final Prediction (R² = {r2:.4f})')
            else:
                plt.title(f'Round {round_num} Final Prediction (R² undefined)')

            min_val = min(pred_df['actual'].min(), pred_df['predicted'].min())
            max_val = max(pred_df['actual'].max(), pred_df['predicted'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)  # Perfect prediction line
            plt.grid(True)

            # Plot 3: EBM Predictions Correlation
            plt.subplot(2, 2, 3)
            plt.scatter(pred_df['actual'], pred_df['ebm_pred'])
            plt.xlabel('Actual')
            plt.ylabel('EBM Prediction')
            if np.var(pred_df['actual']) > 0:
                r2_ebm = r2_score(pred_df['actual'], pred_df['ebm_pred'])
                plt.title(f'EBM Base Prediction (R² = {r2_ebm:.4f})')
            else:
                plt.title('EBM Base Prediction (R² undefined)')
            plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)
            plt.grid(True)

            # Plot 4: Residuals
            plt.subplot(2, 2, 4)
            residuals = pred_df['actual'] - pred_df['ebm_pred']
            plt.scatter(pred_df['actual'], residuals)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel('Actual')
            plt.ylabel('Residuals (Actual - EBM)')
            plt.title('Residuals vs Actual')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(index_output_dir, f"round_{round_num}_plot.png"))
            plt.close()

            successful_rounds += 1

        except Exception as e:
            print(f"Error processing round {round_num}: {str(e)}")
            failed_rounds += 1
            continue
    
    # Check if we have any successful rounds
    if not all_metrics['mae']:
        print(f"No successful rounds for target {target}. All {failed_rounds} rounds failed.")

        # Save a minimal summary
        summary = {
            'processed_rounds': failed_rounds + successful_rounds,
            'successful_rounds': successful_rounds,
            'failed_rounds': failed_rounds,
            'error': "All rounds failed"
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

        # Create overall correlation plots
        plt.figure(figsize=(15, 12))
        
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

        # Plot 3: Model Contribution - Predictions vs Time
        plt.subplot(2, 2, 3)
        # Get a subset of data points to avoid overcrowded plot
        sample_size = min(5000, len(combined_predictions))
        sample_indices = np.linspace(0, len(combined_predictions)-1, sample_size, dtype=int)
        sample_df = combined_predictions.iloc[sample_indices]
        
        plt.stackplot(sample_df.index, 
                     [sample_df['ebm_pred'], sample_df['xgb_pred']], 
                     labels=['EBM Base Prediction', 'XGBoost Residual Correction'],
                     alpha=0.7)
        plt.plot(sample_df.index, sample_df['actual'], 'k-', label='Actual', linewidth=1)
        plt.legend(loc='upper right')
        plt.title('Model Contribution Over Sample Points')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.grid(True)

        # Plot 4: Residual Analysis
        plt.subplot(2, 2, 4)
        final_residuals = overall_actual - overall_predicted
        plt.hist(final_residuals, bins=50)
        plt.title(f'Final Residuals Distribution (Mean: {np.mean(final_residuals):.4f})')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Add summary statistics to plot
        overall_mae = mean_absolute_error(overall_actual, overall_predicted)
        overall_rmse = np.sqrt(mean_squared_error(overall_actual, overall_predicted))
        overall_mae_ebm = mean_absolute_error(overall_actual, overall_ebm_pred)
        overall_rmse_ebm = np.sqrt(mean_squared_error(overall_actual, overall_ebm_pred))

        stats_text = (
            f"Overall Statistics:\n"
            f"Final Model:\n"
            f"  MAE: {overall_mae:.4f}\n"
            f"  RMSE: {overall_rmse:.4f}\n"
            f"  R²: {overall_r2:.4f}\n"
            f"EBM Base Model:\n"
            f"  MAE: {overall_mae_ebm:.4f}\n"
            f"  RMSE: {overall_rmse_ebm:.4f}\n"
            f"  R²: {overall_r2_ebm:.4f}\n"
            f"Samples: {len(overall_actual)}"
        )

        plt.figtext(0.5, 0.01, stats_text, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8), ha='center')

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout for additional text at bottom
        plt.savefig(os.path.join(index_output_dir, "overall_correlation.png"))
        plt.close()
    else:
        overall_r2 = float('nan')
        overall_r2_ebm = float('nan')
        overall_mae = float('nan')
        overall_rmse = float('nan')
        overall_mae_ebm = float('nan')
        overall_rmse_ebm = float('nan')

    # Calculate summary statistics
    summary = {
        'avg_mae': np.mean(all_metrics['mae']),
        'avg_rmse': np.mean(all_metrics['rmse']),
        'std_mae': np.std(all_metrics['mae']),
        'std_rmse': np.std(all_metrics['rmse']),
        'min_mae': np.min(all_metrics['mae']),
        'overall_r2': overall_r2,
        'overall_r2_ebm': overall_r2_ebm,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'overall_mae_ebm': overall_mae_ebm,
        'overall_rmse_ebm': overall_rmse_ebm,
        'processed_rounds': failed_rounds + successful_rounds,
        'successful_rounds': successful_rounds,
        'failed_rounds': failed_rounds
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

    # Create MAE/RMSE summary plots
    if round_results:
        plt.figure(figsize=(12, 5))

        # Plot MAE across rounds
        plt.subplot(1, 2, 1)
        rounds = list(round_results.keys())
        mae_values = [round_results[r]['mae'] for r in rounds]
        plt.bar(rounds, mae_values)
        plt.axhline(y=overall_mae, color='r', linestyle='-', label=f'Overall MAE: {overall_mae:.4f}')
        plt.xlabel('Round Number')
        plt.ylabel('MAE')
        plt.title('MAE Values Across Rounds')
        plt.grid(True, axis='y')
        plt.legend()

        # Plot RMSE across rounds
        plt.subplot(1, 2, 2)
        rmse_values = [round_results[r]['rmse'] for r in rounds]
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
    
    # EBM parameters
    parser.add_argument('--ebm-max-bins', type=int, default=256,
                       help='Max bins for EBM (default: 256)')
    
    parser.add_argument('--ebm-interactions', type=int, default=10,
                       help='Number of interactions for EBM (default: 10)')
    
    parser.add_argument('--ebm-learning-rate', type=float, default=0.01,
                       help='Learning rate for EBM (default: 0.01)')
    
    # XGBoost parameters
    parser.add_argument('--xgb-n-estimators', type=int, default=500,
                       help='Number of estimators for XGBoost (default: 500)')
    
    parser.add_argument('--xgb-learning-rate', type=float, default=0.05,
                       help='Learning rate for XGBoost (default: 0.05)')
    
    parser.add_argument('--xgb-max-depth', type=int, default=6,
                       help='Max depth for XGBoost (default: 6)')
    
    parser.add_argument('--xgb-subsample', type=float, default=0.8,
                       help='Subsample ratio for XGBoost (default: 0.8)')
    
    parser.add_argument('--xgb-colsample-bytree', type=float, default=0.9,
                        help='Colsample by tree for XGBoost (default: 0.9)')
    
    args = parser.parse_args()
    
    # Configure EBM parameters
    ebm_params = {
        'max_bins': args.ebm_max_bins,
        'max_interaction_bins': 32,
        'interactions': args.ebm_interactions,
        'learning_rate': args.ebm_learning_rate,
        'min_samples_leaf': 5,
        'random_state': 42
    }
    
    # Configure XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': args.xgb_n_estimators,
        'learning_rate': args.xgb_learning_rate,
        'max_depth': args.xgb_max_depth,
        'subsample': args.xgb_subsample,
        'colsample_bytree': args.xgb_colsample_bytree,
        'random_state': 42
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
                
                run_cv_for_target_stacked(
                    target=target,
                    start_round=args.start,
                    end_round=args.end,
                    step=args.step,
                    ebm_params=ebm_params,
                    xgb_params=xgb_params,
                    output_dir=args.output,
                    organized_dir=args.organized_dir,
                    target_index=target_index
                )
        except Exception as e:
            print(f"Error processing target {target}: {str(e)}")

if __name__ == "__main__":
    main()