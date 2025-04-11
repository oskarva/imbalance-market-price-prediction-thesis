import os
import argparse
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

def apply_naive_model(y_train, y_test, step_size=32):
    """
    Apply the naive forecasting model.
    
    Args:
        y_train: Training target values (Series or DataFrame column)
        y_test: Test target values (Series or DataFrame column)
        step_size: Number of steps to keep the same prediction (default: 32)
        
    Returns:
        Array of predicted values
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
    
    # Initialize predictions array
    predictions = np.zeros_like(y_test_vals)
    
    # Apply the naive model
    for i in range(len(y_test_vals)):
        if i == 0:
            # For the first prediction, use the last value from the training set
            if len(y_train_vals) > 0:
                predictions[i] = y_train_vals[-1]
            else:
                # If there's no training data, use the first value of the test set
                predictions[i] = y_test_vals[0]
        else:
            # For subsequent predictions, use the value at the last step boundary
            last_step_idx = ((i - 1) // step_size) * step_size
            predictions[i] = y_test_vals[last_step_idx]
    
    return predictions

def train_and_evaluate_naive(X_train, y_train, X_test, y_test, step_size=32):
    """
    Apply naive model and evaluate performance with proper data cleaning.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        step_size: Step size for naive model
        
    Returns:
        Dictionary with evaluation metrics and predictions
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
            'metrics': {
                'mae': float('nan'),
                'rmse': float('nan')
            },
            'predictions': pd.DataFrame(columns=['actual', 'predicted', 'timestamp']),
            'rows_removed': {
                'train': np.sum(~is_valid_train),
                'test': np.sum(~is_valid_test)
            }
        }
    
    # Apply naive model
    y_pred = apply_naive_model(
        y_train=pd.Series(y_train_vals_clean, index=X_train_clean.index), 
        y_test=pd.Series(y_test_vals_clean, index=X_test_clean.index),
        step_size=step_size
    )
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_vals_clean, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_vals_clean, y_pred))
    
    # Create DataFrame with predictions
    pred_df = pd.DataFrame({
        'actual': y_test_vals_clean,
        'predicted': y_pred,
        'timestamp': X_test_clean.index,
        'round': np.full(len(y_test_vals_clean), -1)  # Will be filled with round number by caller
    })
    
    return {
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

def run_cv_for_target_naive(target, start_round=0, end_round=None, step=1,
                          step_size=32, output_dir=None,
                          organized_dir="./src/data/csv", target_index=0):
    """
    Run cross-validation for a specific target with naive model and overall R² calculation.
    
    Args:
        target: Target directory name
        start_round: First CV round to process
        end_round: Last CV round to process (None = all available)
        step: Step size for processing rounds
        step_size: Step size for naive model
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
        index_output_dir = f"./results/naive/{target}_run_{timestamp}_ind_{target_index}"
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
        'step_size': step_size,
        'timestamp': timestamp,
        'target_index': target_index
    }
    
    with open(os.path.join(index_output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nRunning naive model cross-validation for target: {target}, index: {target_index}")
    print(f"Rounds: {start_round} to {end_round-1} with step {step}")
    print(f"Total rounds available: {total_rounds}")
    print(f"Naive model step size: {step_size}")

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

            # Apply naive model and evaluate
            result = train_and_evaluate_naive(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                step_size=step_size
            )

            # Check if we skipped this round due to not enough valid data
            if 'predictions' not in result or len(result['predictions']) == 0:
                print(f"  Skipping round {round_num} due to insufficient valid data.")
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

            # Calculate round-specific R² for the plot title only
            if np.var(pred_df['actual']) > 0:
                r2 = r2_score(pred_df['actual'], pred_df['predicted'])
                plt.title(f'Round {round_num} Correlation (R² = {r2:.4f})')
            else:
                plt.title(f'Round {round_num} Correlation (R² undefined)')

            min_val = min(pred_df['actual'].min(), pred_df['predicted'].min())
            max_val = max(pred_df['actual'].max(), pred_df['predicted'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)  # Perfect prediction line
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

        # Calculate overall R² on all predictions combined
        overall_actual = combined_predictions['actual'].values
        overall_predicted = combined_predictions['predicted'].values

        # Handle potential division by zero in R² calculation
        if np.var(overall_actual) > 0:
            overall_r2 = r2_score(overall_actual, overall_predicted)
        else:
            overall_r2 = float('nan')
            print("Warning: Zero variance in combined actual values, overall R² is undefined")

        # Save combined predictions
        combined_predictions.to_csv(os.path.join(index_output_dir, "all_predictions.csv"), index=False)

        # Create overall correlation plot
        plt.figure(figsize=(10, 8))
        plt.scatter(overall_actual, overall_predicted, alpha=0.5)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        if not np.isnan(overall_r2):
            plt.title(f'Overall Correlation (R² = {overall_r2:.4f})')
        else:
            plt.title('Overall Correlation (R² undefined)')

        min_val = min(overall_actual.min(), overall_predicted.min())
        max_val = max(overall_actual.max(), overall_predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)
        plt.grid(True)

        # Add summary statistics to plot
        overall_mae = mean_absolute_error(overall_actual, overall_predicted)
        overall_rmse = np.sqrt(mean_squared_error(overall_actual, overall_predicted))

        stats_text = (
            f"Overall Statistics:\n"
            f"MAE: {overall_mae:.4f}\n"
            f"RMSE: {overall_rmse:.4f}\n"
            f"R²: {overall_r2:.4f}\n"
            f"Samples: {len(overall_actual)}"
        )

        plt.figtext(0.15, 0.8, stats_text, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(index_output_dir, "overall_correlation.png"))
        plt.close()
    else:
        overall_r2 = float('nan')
        overall_mae = float('nan')
        overall_rmse = float('nan')

    # Calculate summary statistics
    summary = {
        'avg_mae': np.mean(all_metrics['mae']),
        'avg_rmse': np.mean(all_metrics['rmse']),
        'std_mae': np.std(all_metrics['mae']),
        'std_rmse': np.std(all_metrics['rmse']),
        'min_mae': np.min(all_metrics['mae']),
        'overall_r2': overall_r2,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
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
    parser = argparse.ArgumentParser(description='Run cross-validation with naive model and overall R² calculation')
    
    parser.add_argument('--targets', type=str, default=None,
                       help='Target to process, comma-separated (default: show available targets)')
    
    parser.add_argument('--start', type=int, default=0,
                       help='First CV round to process (default: 0)')
    
    parser.add_argument('--end', type=int, default=None,
                       help='Last CV round to process (default: all available)')
    
    parser.add_argument('--step', type=int, default=1,
                       help='Step size for processing rounds (default: 1)')
    
    parser.add_argument('--step-size', type=int, default=32,
                       help='Step size for naive model predictions (default: 32)')
    
    parser.add_argument('--output', type=str, default='./results/naive',
                       help='Base directory to save results (default: ./results/naive)')
    
    parser.add_argument('--organized-dir', type=str, default='./src/data/csv',
                       help='Base directory for organized files (default: ./src/data/csv)')
    
    parser.add_argument('--list', action='store_true',
                       help='List available targets and exit')
    
    parser.add_argument('--target-index', type=int, default=None,
                        help='Index of target to process (0 or 1, default: run both)')
    
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
                run_cv_for_target_naive(
                    target=target,
                    start_round=args.start,
                    end_round=args.end,
                    step=args.step,
                    step_size=args.step_size,
                    output_dir=args.output,
                    organized_dir=args.organized_dir,
                    target_index=target_index
                )
        except Exception as e:
            print(f"Error processing target {target}: {str(e)}")

if __name__ == "__main__":
    main()