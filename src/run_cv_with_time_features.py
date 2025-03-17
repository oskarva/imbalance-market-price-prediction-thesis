"""
Script to run cross-validation using X files with time features.
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

def get_available_targets(organized_dir="./src/data/organized"):
    """
    Get list of available targets from the organized directory structure.
    
    Args:
        organized_dir: Base directory for organized files
        
    Returns:
        List of available targets (directory names)
    """
    targets = []
    
    # Look for subdirectories that contain y_train and y_test files
    for item in os.listdir(organized_dir):
        dir_path = os.path.join(organized_dir, item)
        
        # Skip if not a directory or if it's an X_files directory
        if not os.path.isdir(dir_path) or item.startswith('X_files'):
            continue
        
        # Check if directory contains y_train files
        if any(f.startswith('y_train_') for f in os.listdir(dir_path)):
            targets.append(item)
    
    return targets

def load_cv_round(cv_round, target_dir, x_files_dir):
    """
    Load a specific cross-validation round's data for a target.
    
    Args:
        cv_round: The cross-validation round number to load
        target_dir: Directory containing target-specific files
        x_files_dir: Directory containing X files with time features
        
    Returns:
        Tuple containing (X_train, y_train, X_test, y_test) as pandas DataFrames
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
        
        return X_train, y_train, X_test, y_test
    
    except FileNotFoundError as e:
        print(f"Could not load cross-validation round {cv_round}: {str(e)}")
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

def train_and_evaluate(X_train, y_train, X_test, y_test, params):
    """
    Train model and evaluate performance with proper data cleaning.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        params: Model parameters
        
    Returns:
        Dictionary with evaluation metrics and model
    """
    # Extract target values as 1D arrays
    y_train_vals = y_train.iloc[:, 0].values
    y_test_vals = y_test.iloc[:, 0].values
    
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
                'rmse': float('nan')
            },
            'predictions': pd.DataFrame(columns=['actual', 'predicted', 'timestamp']),
            'rows_removed': {
                'train': np.sum(~is_valid_train),
                'test': np.sum(~is_valid_test)
            }
        }
    
    # Initialize and train model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_clean, y_train_vals_clean)
    
    # Make predictions
    y_pred = model.predict(X_test_clean)
    
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
    
    # Calculate top feature importances
    feature_importances = model.feature_importances_
    feature_names = X_train_clean.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'metrics': {
            'mae': mae,
            'rmse': rmse
        },
        'predictions': pred_df,
        'rows_removed': {
            'train': np.sum(~is_valid_train),
            'test': np.sum(~is_valid_test)
        },
        'feature_importances': importance_df.head(20).to_dict('records')
    }

def run_cv_for_target(target, start_round=0, end_round=None, step=1,
                     model_params=None, output_dir=None,
                     organized_dir="./src/data/organized",
                     use_time_features=True):
    """
    Run cross-validation for a specific target with overall R² calculation.
    
    Args:
        target: Target directory name
        start_round: First CV round to process
        end_round: Last CV round to process (None = all available)
        step: Step size for processing rounds
        model_params: Parameters for the model
        output_dir: Directory to save results
        organized_dir: Base directory for organized files
        use_time_features: Whether to use X files with time features
        
    Returns:
        Dictionary with results summary
    """
    # Set up paths
    target_dir = os.path.join(organized_dir, target)
    
    # Select appropriate X files directory based on use_time_features flag
    if use_time_features:
        x_files_dir = os.path.join(organized_dir, "X_files_with_time_features")
    else:
        x_files_dir = os.path.join(organized_dir, "X_files")
    
    # Check if target directory exists
    if not os.path.isdir(target_dir):
        raise ValueError(f"Target directory not found: {target_dir}")
    
    # Check if X files directory exists
    if not os.path.isdir(x_files_dir):
        raise ValueError(f"X files directory not found: {x_files_dir}")
    
    # Default model parameters if none provided
    if model_params is None:
        model_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,  # Increased from 100
            'learning_rate': 0.05,  # Increased from 0.01
            'max_depth': 6,  # Increased from 3
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'random_state': 42
        }
    
    # Set up output directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        time_features_str = "with_time_features" if use_time_features else "no_time_features"
        output_dir = f"./results/{target}_{time_features_str}_run_{timestamp}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total number of rounds
    total_rounds = get_cv_round_count(target_dir)
    
    # Set default end_round if not provided
    if end_round is None:
        end_round = total_rounds
    else:
        end_round = min(end_round, total_rounds)
        
    # Save configuration
    config = {
        'target': target,
        'start_round': start_round,
        'end_round': end_round,
        'step': step,
        'model_params': model_params,
        'timestamp': timestamp,
        'use_time_features': use_time_features
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nRunning cross-validation for target: {target}")
    print(f"Using {'time features' if use_time_features else 'regular features'}")
    print(f"Output directory: {output_dir}")
    print(f"Rounds: {start_round} to {end_round-1} with step {step}")
    print(f"Total rounds available: {total_rounds}")
    print(f"Model parameters: {model_params}")
    
    # Track metrics across all rounds
    all_metrics = {
        'mae': [],
        'rmse': []
    }
    
    # Track all predictions for overall R² calculation
    all_predictions = []
    
    # Store feature importance across rounds
    all_feature_importances = {}
    
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
                x_files_dir=x_files_dir
            )
            
            # Train and evaluate model
            result = train_and_evaluate(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                params=model_params
            )
            
            # Check if we skipped this round due to not enough valid data
            if result['model'] is None:
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
            pred_df.to_csv(os.path.join(output_dir, f"round_{round_num}_predictions.csv"), index=False)
            
            # Save and aggregate feature importances
            if 'feature_importances' in result:
                # Save this round's feature importances
                importances_df = pd.DataFrame(result['feature_importances'])
                importances_df.to_csv(os.path.join(output_dir, f"round_{round_num}_feature_importances.csv"), index=False)
                
                # Aggregate feature importances across rounds
                for item in result['feature_importances']:
                    feature = item['feature']
                    importance = item['importance']
                    
                    if feature not in all_feature_importances:
                        all_feature_importances[feature] = []
                    
                    all_feature_importances[feature].append(importance)
            
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
            plt.savefig(os.path.join(output_dir, f"round_{round_num}_plot.png"))
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
        
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        return summary
    
    # Calculate average feature importances
    if all_feature_importances:
        avg_importances = []
        for feature, importances in all_feature_importances.items():
            avg_importances.append({
                'feature': feature,
                'avg_importance': np.mean(importances),
                'std_importance': np.std(importances),
                'appearance_count': len(importances),
                'appearance_pct': 100 * len(importances) / successful_rounds
            })
        
        # Sort by average importance
        avg_importances.sort(key=lambda x: x['avg_importance'], reverse=True)
        
        # Save to CSV
        pd.DataFrame(avg_importances).to_csv(os.path.join(output_dir, "feature_importances.csv"), index=False)
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        
        # Get top 20 features
        top_features = avg_importances[:20]
        
        # Plot importance
        plt.barh([x['feature'] for x in top_features], 
                [x['avg_importance'] for x in top_features])
        plt.xlabel('Average Importance')
        plt.title('Top 20 Feature Importances')
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importances.png"))
        plt.close()
    
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
        combined_predictions.to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False)
        
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
        plt.savefig(os.path.join(output_dir, "overall_correlation.png"))
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
        'failed_rounds': failed_rounds,
        'used_time_features': use_time_features
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
        plt.savefig(os.path.join(output_dir, "metrics_summary.png"))
        plt.close()
    
    # Save summary to file
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        # Convert numpy values to regular Python types for JSON serialization
        summary_json = {k: float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v) else 
                          None if isinstance(v, (np.floating, float)) and np.isnan(v) else v 
                          for k, v in summary.items()}
        json.dump(summary_json, f, indent=4)
    
    # Save individual round results
    with open(os.path.join(output_dir, 'round_results.json'), 'w') as f:
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
    parser = argparse.ArgumentParser(description='Run cross-validation with time features')
    
    parser.add_argument('--target', type=str, default=None,
                       help='Target to process (default: show available targets)')
    
    parser.add_argument('--start', type=int, default=0,
                       help='First CV round to process (default: 0)')
    
    parser.add_argument('--end', type=int, default=None,
                       help='Last CV round to process (default: all available)')
    
    parser.add_argument('--step', type=int, default=1,
                       help='Step size for processing rounds (default: 1)')
    
    parser.add_argument('--output', type=str, default='./results',
                       help='Base directory to save results (default: ./results)')
    
    parser.add_argument('--organized-dir', type=str, default='./src/data/organized',
                       help='Base directory for organized files (default: ./src/data/organized)')
    
    parser.add_argument('--n_estimators', type=int, default=500,
                       help='Number of estimators for XGBoost (default: 500)')
    
    parser.add_argument('--learning_rate', type=float, default=0.05,
                       help='Learning rate for XGBoost (default: 0.05)')
    
    parser.add_argument('--max_depth', type=int, default=6,
                       help='Max depth for XGBoost (default: 6)')
    
    parser.add_argument('--subsample', type=float, default=0.9,
                       help='Subsample ratio for XGBoost (default: 0.9)')
    
    parser.add_argument('--no-time-features', action='store_true',
                       help='Use regular X files without time features')
    
    parser.add_argument('--list', action='store_true',
                       help='List available targets and exit')
    
    args = parser.parse_args()
    
    # Whether to use time features or not
    use_time_features = not args.no_time_features
    
    # Check if time features directory exists
    if use_time_features:
        time_features_dir = os.path.join(args.organized_dir, "X_files_with_time_features")
        if not os.path.isdir(time_features_dir):
            print(f"Warning: Time features directory not found: {time_features_dir}")
            print("Please run add_time_features.py first to create time feature files.")
            print("Falling back to regular X files.")
            use_time_features = False
    
    # Get available targets
    available_targets = get_available_targets(args.organized_dir)
    
    if args.list or not available_targets or args.target is None:
        print("\nAvailable targets:")
        for target in available_targets:
            target_dir = os.path.join(args.organized_dir, target)
            try:
                num_rounds = get_cv_round_count(target_dir)
                print(f"  {target} ({num_rounds} rounds)")
            except Exception as e:
                print(f"  {target} (Error: {str(e)})")
        
        if not args.target:
            return
    
    # Configure model parameters
    model_params = {
        'objective': 'reg:squarederror',
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth,
        'subsample': args.subsample,
        'colsample_bytree': 0.9,  # Default value
        'random_state': 42
    }
    
    # Run for a specific target if provided, otherwise run for all available targets
    if args.target:
        if args.target not in available_targets:
            print(f"Target '{args.target}' not found in available targets.")
            print("Available targets:")
            for target in available_targets:
                print(f"  {target}")
            return
        
        # Run CV for the specified target
        time_features_str = "with_time_features" if use_time_features else "no_time_features"
        output_dir = os.path.join(args.output, f"{args.target}_{time_features_str}_run_{time.strftime('%Y%m%d-%H%M%S')}")
        
        run_cv_for_target(
            target=args.target,
            start_round=args.start,
            end_round=args.end,
            step=args.step,
            model_params=model_params,
            output_dir=output_dir,
            organized_dir=args.organized_dir,
            use_time_features=use_time_features
        )
    else:
        # Run for all available targets
        for target in available_targets:
            try:
                print(f"\n{'='*50}")
                print(f"Processing target: {target}")
                print(f"{'='*50}")
                
                time_features_str = "with_time_features" if use_time_features else "no_time_features"
                output_dir = os.path.join(args.output, f"{target}_{time_features_str}_run_{time.strftime('%Y%m%d-%H%M%S')}")
                
                run_cv_for_target(
                    target=target,
                    start_round=args.start,
                    end_round=args.end,
                    step=args.step,
                    model_params=model_params,
                    output_dir=output_dir,
                    organized_dir=args.organized_dir,
                    use_time_features=use_time_features
                )
            except Exception as e:
                print(f"Error processing target {target}: {str(e)}")

if __name__ == "__main__":
    main()