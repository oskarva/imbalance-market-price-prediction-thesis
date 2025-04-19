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
import joblib
from functools import partial
import warnings # Added warning import

# Suppress warnings (optional)
warnings.filterwarnings('ignore')

PREDEFINED_PARAMETER_SETS = {
        'set1_robust': {
            'objective': 'reg:absoluteerror',
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'max_depth': 6,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 5,
            'random_state': 42,
            'tree_method': 'hist'
        },
        'set2_regularized': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'n_estimators': 500,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1.0,
            'random_state': 42,
            'tree_method': 'hist'
        },
        'set3_outlier_robust': {
            'objective': 'reg:pseudohubererror',
            'learning_rate': 0.03,
            'n_estimators': 700,
            'max_depth': 5,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': 42,
            'tree_method': 'hist'
        },
        'set4_extreme_events': {
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.5,  # Median regression as base
            'learning_rate': 0.02,
            'n_estimators': 1200,
            'max_depth': 7,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 3,
            'random_state': 42,
            'tree_method': 'hist'
        },
        'set5_ensemble_diverse': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.01,
            'n_estimators': 800,
            'max_depth': 5,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'colsample_bylevel': 0.75,  # Additional randomization
            'gamma': 0.5,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'random_state': 42,
            'tree_method': 'hist'
        }
    }



def add_time_features(df):
    """
    Add time-based features to the dataframe.

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with additional time features
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_with_features = df.copy()

    # Ensure index is datetime
    if not isinstance(df_with_features.index, pd.DatetimeIndex):
        print("Warning: Index is not DatetimeIndex. Attempting conversion.")
        try:
            df_with_features.index = pd.to_datetime(df_with_features.index, utc=True)
        except Exception as e:
            print(f"Error converting index to datetime: {e}")
            # Return original df if conversion fails to prevent further errors
            return df

    # Time of day features (cyclic encoding to handle midnight/noon transitions)
    hours_in_day = 24
    df_with_features['sin_hour'] = np.sin(2 * np.pi * df_with_features.index.hour / hours_in_day)
    df_with_features['cos_hour'] = np.cos(2 * np.pi * df_with_features.index.hour / hours_in_day)

    # Month features (cyclic encoding to handle year transitions)
    months_in_year = 12
    df_with_features['sin_month'] = np.sin(2 * np.pi * df_with_features.index.month / months_in_year)
    df_with_features['cos_month'] = np.cos(2 * np.pi * df_with_features.index.month / months_in_year)

    return df_with_features


def get_available_targets(organized_dir="./src/data/csv", areas=["no1", "no2", "no3", "no4", "no5"], phase="validation"):
    """
    Get list of available targets from the organized directory structure.

    Args:
        organized_dir: Base directory for organized files
        areas: List of area directories to check
        phase: Either "validation" or "test" to specify which phase's data to check

    Returns:
        List of available targets (directory names)
    """
    targets = []

    # Look for subdirectories that contain validation_rounds or test_rounds folders
    for item in os.listdir(organized_dir):
        if item in areas:
            # Check if this area has the specified phase's folder
            phase_dir = os.path.join(organized_dir, item, f"{phase}_rounds")
            if os.path.isdir(phase_dir) and any(f.startswith('y_test_') for f in os.listdir(phase_dir)):
                targets.append(item)

    return targets

def load_cv_round(cv_round, target_dir, x_files_dir, target_index=0, phase="validation"):
    """
    Load a specific cross-validation round's data for a target,
    adding time features to X data.

    Args:
        cv_round: The cross-validation round number to load
        target_dir: Directory containing target-specific files
        x_files_dir: Directory containing X files
        target_index: Index (0-based) for which target column to return if multiple targets exist
        phase: Either "validation" or "test" to specify which phase's data to load

    Returns:
        Tuple containing (X_train, y_train, X_test, y_test) as pandas DataFrames or Series.
        X_train and X_test will have added time features.
        y_train and y_test will be the selected target column.
    """
    try:
        # Construct paths with phase folder
        phase_folder = f"{phase}_rounds"
        phase_dir = os.path.join(target_dir, phase_folder)
        x_phase_dir = os.path.join(x_files_dir, phase_folder)

        # Verify directories exist
        if not os.path.isdir(phase_dir):
            raise FileNotFoundError(f"Phase directory not found: {phase_dir}")
        if not os.path.isdir(x_phase_dir):
            raise FileNotFoundError(f"X phase directory not found: {x_phase_dir}")

        # Load X data from phase directory
        X_train = pd.read_csv(f"{x_phase_dir}/X_train_{cv_round}.csv", index_col=0)
        X_test = pd.read_csv(f"{x_phase_dir}/X_test_{cv_round}.csv", index_col=0)

        # Load y data from phase directory
        y_train = pd.read_csv(f"{phase_dir}/y_train_{cv_round}.csv", index_col=0)
        y_test = pd.read_csv(f"{phase_dir}/y_test_{cv_round}.csv", index_col=0)

        # Convert indices to datetime with utc=True to avoid warnings
        X_train.index = pd.to_datetime(X_train.index, utc=True)
        X_test.index = pd.to_datetime(X_test.index, utc=True)
        y_train.index = pd.to_datetime(y_train.index, utc=True)
        y_test.index = pd.to_datetime(y_test.index, utc=True)

        # --- Add Time Features to X data ---
        # print(f"  Adding time features to X_train and X_test for round {cv_round}...") # Less verbose
        X_train = add_time_features(X_train)
        X_test = add_time_features(X_test)
        # print(f"  New X_train columns: {X_train.columns.tolist()}") # Less verbose
        # print(f"  New X_test columns: {X_test.columns.tolist()}") # Less verbose
        # --- End of Adding Time Features ---

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
        print(f"Error loading CV round {cv_round} for phase {phase}: {e}")
        raise

def get_cv_round_count(target_dir, phase="validation"):
    """
    Get the total number of cross-validation rounds available for a target.

    Args:
        target_dir: Directory containing target-specific files
        phase: Either "validation" or "test" to specify which phase's data to count

    Returns:
        int: The number of cross-validation rounds
    """
    # Construct path with phase folder
    phase_folder = f"{phase}_rounds"
    phase_dir = os.path.join(target_dir, phase_folder)

    # Verify directory exists
    if not os.path.isdir(phase_dir):
        raise FileNotFoundError(f"Phase directory not found: {phase_dir}")

    y_test_files = [f for f in os.listdir(phase_dir) if f.startswith('y_test_')]

    # Extract round numbers from filenames
    round_numbers = []
    for f in y_test_files:
        try:
            round_num = int(f.split('_')[-1].split('.')[0])
            round_numbers.append(round_num)
        except (ValueError, IndexError):
            continue

    if not round_numbers:
        raise FileNotFoundError(f"No cross-validation files found in {phase_dir}")

    return max(round_numbers) + 1  # +1 because we count from 0

def train_and_evaluate(X_train, y_train, X_test, y_test, params):
    """
    Train model and evaluate performance with proper data cleaning.
    Assumes X_train and X_test already have necessary features (including time features).

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        params: Model parameters

    Returns:
        Dictionary with evaluation metrics and model
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
        # Ensure X_train indices match y_train indices before filtering
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
        # Ensure X_test indices match y_test indices before filtering
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

    # Initialize and train model
    model = xgb.XGBRegressor(**params)
    print(f"  Training model with {X_train_clean.shape[1]} features...")
    model.fit(X_train_clean, y_train_vals_clean)
    # Save model after training
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', 'xgb_last_run.joblib'), compress=3)

    # Make predictions
    print(f"  Making predictions on test set with {X_test_clean.shape[1]} features...")
    y_pred = model.predict(X_test_clean)

    # Calculate metrics
    mae = mean_absolute_error(y_test_vals_clean, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_vals_clean, y_pred))
    # Calculate R² only if variance is sufficient
    r2 = float('nan')
    if np.var(y_test_vals_clean) > 1e-9:
        r2 = r2_score(y_test_vals_clean, y_pred)


    # Create DataFrame with predictions
    pred_df = pd.DataFrame({
        'actual': y_test_vals_clean,
        'predicted': y_pred,
        'timestamp': X_test_clean.index, # Use index from cleaned X_test
        'round': np.full(len(y_test_vals_clean), -1)  # Will be filled with round number by caller
    })

    return {
        'model': model,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2 
        },
        'predictions': pred_df,
        'rows_removed': {
            'train': np.sum(~is_valid_train),
            'test': np.sum(~is_valid_test)
        }
    }

def run_cv_for_target(target, start_round=0, end_round=None, step=1,
                     model_params=None, output_dir=None,
                     organized_dir="./src/data/csv", target_index=0, phase="validation",
                     param_set_name=None): 
    """
    Run cross-validation for a specific target with overall R² calculation.

    Args:
        target: Target directory name
        start_round: First CV round to process
        end_round: Last CV round to process (None = all available)
        step: Step size for processing rounds
        model_params: Parameters for the model (used if param_set_name is None)
        output_dir: Directory to save results
        organized_dir: Base directory for organized files
        target_index: Index of target variable (0 or 1) for regulation up/down
        phase: Either "validation" or "test" to specify which phase's data to use
        param_set_name: Name of the parameter set (if provided)

    Returns:
        Dictionary with results summary
    """
    # Set up paths
    target_dir = os.path.join(organized_dir, target)
    x_files_dir = os.path.join(organized_dir, target) # Assuming X files are in the same dir

    # Check if target directory exists
    if not os.path.isdir(target_dir):
        raise ValueError(f"Target directory not found: {target_dir}")

    # Determine model parameters (prioritize param_set_name)
    if param_set_name:
        if param_set_name in PREDEFINED_PARAMETER_SETS:
            actual_model_params = PREDEFINED_PARAMETER_SETS[param_set_name]
            print(f"Using predefined parameter set: '{param_set_name}'")
        else:
            raise ValueError(f"Parameter set name '{param_set_name}' not found in PREDEFINED_PARAMETER_SETS.")
    elif model_params:
        actual_model_params = model_params
        param_set_name = "custom" # Indicate custom params were used
        print("Using custom parameters provided via command line.")

    # Set up output directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Get total number of rounds
    total_rounds = get_cv_round_count(target_dir, phase=phase)

    # Set default end_round if not provided
    if end_round is None:
        end_round = total_rounds
    else:
        end_round = min(end_round, total_rounds)

    # Create a unique directory for the run
    if output_dir is None:
        # Use the new directory structure including param set name
        base_dir = f"./results/xgboost_timefeat/{timestamp}_{phase}"
        os.makedirs(base_dir, exist_ok=True)
        # Include param_set_name in the directory path
        index_output_dir = f"{base_dir}/{target}_ind_{target_index}_{param_set_name}"
    else:
        # Use the provided output directory directly
        index_output_dir = output_dir
        # Optionally append param_set_name if desired even with explicit output dir
        # index_output_dir = os.path.join(output_dir, f"{target}_ind_{target_index}_{param_set_name}")


    # Create output directory
    os.makedirs(index_output_dir, exist_ok=True)

    # Save configuration
    config = {
        'target': target,
        'start_round': start_round,
        'end_round': end_round,
        'step': step,
        'parameter_set_name': param_set_name, # Store the name used
        'model_params': actual_model_params, # Store the actual params used
        'timestamp': timestamp,
        'target_index': target_index,
        'time_features_added': True, # Indicate time features were used
        'phase': phase  # Save which phase we're using
    }

    with open(os.path.join(index_output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nRunning cross-validation for target: {target}, index: {target_index} (with time features)")
    print(f"Phase: {phase}")
    print(f"Rounds: {start_round} to {end_round-1} with step {step}")
    print(f"Total rounds available: {total_rounds}")
    print(f"Parameter set: '{param_set_name}'")
    print(f"Model parameters: {actual_model_params}")


    # Track metrics across all rounds
    all_metrics = {
        'mae': [],
        'rmse': [],
        'r2': [] 
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
            # Load data for this round (includes adding time features)
            X_train, y_train, X_test, y_test = load_cv_round(
                cv_round=round_num,
                target_dir=target_dir,
                x_files_dir=x_files_dir,
                target_index=target_index,
                phase=phase  # Pass the phase parameter
            )

            # Train and evaluate model
            result = train_and_evaluate(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                params=actual_model_params # Use the determined params
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

            # Update tracking metrics, handling potential NaN for R2
            all_metrics['mae'].append(result['metrics']['mae'])
            all_metrics['rmse'].append(result['metrics']['rmse'])
            if pd.notna(result['metrics']['r2']): # Only append valid R2 scores
                 all_metrics['r2'].append(result['metrics']['r2'])

            # Add to all predictions for overall R² calculation
            all_predictions.append(result['predictions'])

            # Print round results
            print(f"  MAE: {result['metrics']['mae']:.4f}")
            print(f"  RMSE: {result['metrics']['rmse']:.4f}")
            print(f"  R²: {result['metrics']['r2']:.4f}") 

            if 'rows_removed' in result:
                print(f"  Rows removed: {result['rows_removed']['train']} train, {result['rows_removed']['test']} test")

            # Save predictions
            pred_df = result['predictions']
            pred_df.to_csv(os.path.join(index_output_dir, f"round_{round_num}_predictions.csv"), index=False)

            # Create and save plot
            plt.figure(figsize=(12, 6))

            # Plot actual vs predicted values
            plt.subplot(1, 2, 1)
            # Ensure timestamps are sorted for plotting if needed
            pred_df_sorted = pred_df.sort_values('timestamp')
            plt.plot(pred_df_sorted['timestamp'], pred_df_sorted['actual'], 'b-', label='Actual', alpha=0.7)
            plt.plot(pred_df_sorted['timestamp'], pred_df_sorted['predicted'], 'r--', label='Predicted', alpha=0.7)
            plt.title(f'Round {round_num} ({phase}) - Actual vs Predicted')
            plt.ylabel('Value') # Changed from Price for generality
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)

            # Scatter plot to visualize correlation
            plt.subplot(1, 2, 2)
            plt.scatter(pred_df['actual'], pred_df['predicted'], alpha=0.5)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')

            # Use the round-specific R² calculated in train_and_evaluate
            round_r2 = result['metrics']['r2']
            if pd.notna(round_r2):
                plt.title(f'Round {round_num} ({phase}) Correlation (R² = {round_r2:.4f})')
            else:
                plt.title(f'Round {round_num} ({phase}) Correlation (R² undefined)')


            # Add perfect prediction line only if data exists
            if not pred_df.empty:
                min_val = min(pred_df['actual'].min(), pred_df['predicted'].min())
                max_val = max(pred_df['actual'].max(), pred_df['predicted'].max())
                # Check if min/max are valid numbers
                if pd.notna(min_val) and pd.notna(max_val):
                    plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)

            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(index_output_dir, f"round_{round_num}_plot.png"))
            plt.close()

            successful_rounds += 1

        except FileNotFoundError as fnf_error:
            print(f"Error processing round {round_num}: Data file not found - {fnf_error}. Skipping round.")
            failed_rounds += 1
            continue
        except Exception as e:
            print(f"Error processing round {round_num}: {type(e).__name__} - {str(e)}")
            # Optionally add traceback here for debugging
            # import traceback
            # traceback.print_exc()
            failed_rounds += 1
            continue

    # Check if we have any successful rounds
    if not all_metrics['mae']:
        print(f"No successful rounds for target {target}, index {target_index}, phase {phase}. All {failed_rounds} rounds failed or were skipped.")

        # Save a minimal summary
        summary = {
            'processed_rounds': failed_rounds + successful_rounds,
            'successful_rounds': successful_rounds,
            'failed_rounds': failed_rounds,
            'error': "All rounds failed or were skipped",
            'phase': phase,
            'parameter_set_name': param_set_name
        }

        with open(os.path.join(index_output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        return summary

    # Combine all predictions for overall R² calculation
    overall_r2 = float('nan')
    overall_mae = float('nan')
    overall_rmse = float('nan')
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)

        # Calculate overall metrics on all predictions combined
        overall_actual = combined_predictions['actual'].values
        overall_predicted = combined_predictions['predicted'].values

        # Handle potential division by zero in R² calculation
        if np.var(overall_actual) > 1e-9: # Avoid division by zero or near-zero
            overall_r2 = r2_score(overall_actual, overall_predicted)
            overall_mae = mean_absolute_error(overall_actual, overall_predicted)
            overall_rmse = np.sqrt(mean_squared_error(overall_actual, overall_predicted))
        else:
            print("Warning: Near-zero variance in combined actual values, overall R²/MAE/RMSE might be unreliable or undefined")

        # Save combined predictions
        combined_predictions.to_csv(os.path.join(index_output_dir, "all_predictions.csv"), index=False)

        # Create overall correlation plot
        plt.figure(figsize=(10, 8))
        plt.scatter(overall_actual, overall_predicted, alpha=0.3) # Reduced alpha
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        if not np.isnan(overall_r2):
            plt.title(f'Overall Correlation - {phase.capitalize()} Phase (R² = {overall_r2:.4f})')
        else:
            plt.title(f'Overall Correlation - {phase.capitalize()} Phase (R² undefined or unreliable)')

        # Add perfect prediction line only if data exists and is valid
        if len(overall_actual) > 0 and pd.notna(overall_actual.min()) and pd.notna(overall_predicted.min()):
             min_val = min(overall_actual.min(), overall_predicted.min())
             max_val = max(overall_actual.max(), overall_predicted.max())
             if pd.notna(min_val) and pd.notna(max_val):
                 plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)
        plt.grid(True)

        # Add summary statistics to plot
        stats_text = (
            f"Overall Statistics ({phase.capitalize()}):\n"
            f"MAE: {overall_mae:.4f}\n"
            f"RMSE: {overall_rmse:.4f}\n"
            f"R²: {overall_r2:.4f}\n"
            f"Samples: {len(overall_actual)}\n"
            f"Param Set: {param_set_name}" 
        )

        plt.figtext(0.15, 0.75, stats_text, fontsize=12, # Adjusted position
                   bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(index_output_dir, "overall_correlation.png"))
        plt.close()

    # Calculate summary statistics
    summary = {
        'avg_mae': np.mean(all_metrics['mae']) if all_metrics['mae'] else float('nan'),
        'avg_rmse': np.mean(all_metrics['rmse']) if all_metrics['rmse'] else float('nan'),
        'avg_r2': np.mean(all_metrics['r2']) if all_metrics['r2'] else float('nan'), # Avg R2
        'std_mae': np.std(all_metrics['mae']) if all_metrics['mae'] else float('nan'),
        'std_rmse': np.std(all_metrics['rmse']) if all_metrics['rmse'] else float('nan'),
        'std_r2': np.std(all_metrics['r2']) if all_metrics['r2'] else float('nan'), # Std R2
        'min_mae': np.min(all_metrics['mae']) if all_metrics['mae'] else float('nan'),
        'overall_r2': overall_r2,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'processed_rounds': failed_rounds + successful_rounds,
        'successful_rounds': successful_rounds,
        'failed_rounds': failed_rounds,
        'phase': phase,
        'parameter_set_name': param_set_name 
    }

    # Print summary
    print("\nCross-validation summary:")
    print(f"Phase: {phase}")
    print(f"Parameter Set: '{param_set_name}'")
    print(f"Processed {summary['processed_rounds']} rounds ({summary['successful_rounds']} successful, {summary['failed_rounds']} failed/skipped)")
    print(f"Avg MAE: {summary['avg_mae']:.4f} ± {summary['std_mae']:.4f}")
    print(f"Avg RMSE: {summary['avg_rmse']:.4f} ± {summary['std_rmse']:.4f}")
    print(f"Avg R²: {summary['avg_r2']:.4f} ± {summary['std_r2']:.4f}")
    print(f"Overall R² (calculated on all predictions): {summary['overall_r2']:.4f}")
    print(f"Overall MAE: {summary['overall_mae']:.4f}")
    print(f"Overall RMSE: {summary['overall_rmse']:.4f}")

    # Create MAE/RMSE/R2 summary plots
    if round_results:
        plt.figure(figsize=(18, 5)) # Wider figure
        rounds = sorted(list(round_results.keys())) # Sort rounds for plotting
        mae_values = [round_results[r]['mae'] for r in rounds]
        rmse_values = [round_results[r]['rmse'] for r in rounds]
        r2_values = [round_results[r]['r2'] for r in rounds] # Get R2 values


        # Plot MAE across rounds
        plt.subplot(1, 3, 1) # Changed to 1x3 grid
        plt.bar([str(r) for r in rounds], mae_values) # Use string labels for rounds
        if not np.isnan(overall_mae):
             plt.axhline(y=overall_mae, color='r', linestyle='-', label=f'Overall MAE: {overall_mae:.4f}')
        plt.xlabel('Round Number')
        plt.ylabel('MAE')
        plt.title(f'MAE Values Across Rounds - {phase.capitalize()} Phase')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        plt.legend()

        # Plot RMSE across rounds
        plt.subplot(1, 3, 2) # Changed to 1x3 grid
        plt.bar([str(r) for r in rounds], rmse_values) # Use string labels for rounds
        if not np.isnan(overall_rmse):
             plt.axhline(y=overall_rmse, color='r', linestyle='-', label=f'Overall RMSE: {overall_rmse:.4f}')
        plt.xlabel('Round Number')
        plt.ylabel('RMSE')
        plt.title(f'RMSE Values Across Rounds - {phase.capitalize()} Phase')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        plt.legend()

        # Plot R² across rounds
        plt.subplot(1, 3, 3) # Changed to 1x3 grid
        plt.bar([str(r) for r in rounds], r2_values) # Use string labels for rounds
        if not np.isnan(overall_r2):
             plt.axhline(y=overall_r2, color='r', linestyle='-', label=f'Overall R²: {overall_r2:.4f}')
        plt.xlabel('Round Number')
        plt.ylabel('R²')
        plt.title(f'R² Values Across Rounds - {phase.capitalize()} Phase')
        plt.xticks(rotation=45, ha='right')
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
                k: float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v) else
                   None if isinstance(v, (np.floating, float)) and np.isnan(v) else v
                for k, v in metrics.items()
            }
        json.dump(round_results_json, f, indent=4)

    return {
        'summary': summary,
        'round_results': round_results
    }


def try_parameter_sets_for_target(target, start_round=0, end_round=None, step=1,
                               organized_dir="./src/data/csv", target_index=0, phase="validation"):
    """
    Try different predefined parameter sets for a specific target, running full cross-validation for each.
    This will use the modified run_cv_for_target which adds time features.

    Args:
        target: Target directory name
        start_round: First CV round to process
        end_round: Last CV round to process (None = all available)
        step: Step size for processing rounds
        organized_dir: Base directory for organized files
        target_index: Index of target variable (0 or 1) for regulation up/down
        phase: Either "validation" or "test" to specify which phase's data to use

    Returns:
        Dictionary with results summary for each parameter set
    """
    # Use the globally defined parameter sets
    parameter_sets_to_try = PREDEFINED_PARAMETER_SETS

    # Get timestamp for the run
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Set up base output directory
    base_output_dir = f"./results/xgboost_timefeat_optimize/{timestamp}_{phase}"
    os.makedirs(base_output_dir, exist_ok=True)

    # Store results for each parameter set
    optimization_results = {}

    # Try each parameter set
    for param_name, params in parameter_sets_to_try.items():
        print(f"\n{'='*50}")
        print(f"Optimizing: Trying parameter set: {param_name} for target: {target}, index: {target_index}, phase: {phase} (with time features)")
        # print(f"Parameters: {params}") # Already printed inside run_cv_for_target
        print(f"{'='*50}")

        # Define output dir for this specific param set run within the optimization folder
        output_dir_param_set = os.path.join(base_output_dir, f"{target}_ind_{target_index}_{param_name}")


        # Run cross-validation using the parameter set name
        result = run_cv_for_target(
            target=target,
            start_round=start_round,
            end_round=end_round,
            step=step,
            model_params=None, # Pass None for params, use name instead
            output_dir=output_dir_param_set, # Use specific dir for this run
            organized_dir=organized_dir,
            target_index=target_index,
            phase=phase,
            param_set_name=param_name # Pass the name
        )

        # Store the result using the param_name as the key
        optimization_results[param_name] = {
            'parameters': params, # Store params for reference
            'summary': result['summary'] if 'summary' in result else result
        }

    # Create a comparison summary
    comparison = {
        'target': target,
        'target_index': target_index,
        'phase': phase,
        'timestamp': timestamp,
        'parameter_sets_evaluated': list(parameter_sets_to_try.keys()),
        'metrics': {},
        'time_features_added': True
    }

    # Extract key metrics for comparison
    for param_name, result in optimization_results.items():
        if 'summary' in result and isinstance(result['summary'], dict):
            summary = result['summary']
            comparison['metrics'][param_name] = {
                'overall_r2': summary.get('overall_r2', float('nan')),
                'overall_mae': summary.get('overall_mae', float('nan')),
                'overall_rmse': summary.get('overall_rmse', float('nan')),
                'avg_r2': summary.get('avg_r2', float('nan')) # Added avg R2
            }

    # Determine best parameter set based on overall R²
    if comparison['metrics']:
        best_r2 = -float('inf')
        best_param_set = None

        for param_name, metrics in comparison['metrics'].items():
            r2 = metrics.get('overall_r2', -float('inf'))
            # Check if r2 is not nan and greater than current best
            if pd.notna(r2) and r2 > best_r2:
                best_r2 = r2
                best_param_set = param_name

        comparison['best_parameter_set'] = best_param_set
        comparison['best_overall_r2'] = best_r2 if best_param_set is not None else float('nan')

        # Also find best based on average R2 across folds (might be more robust)
        best_avg_r2 = -float('inf')
        best_param_set_avg = None
        for param_name, metrics in comparison['metrics'].items():
            avg_r2 = metrics.get('avg_r2', -float('inf'))
            if pd.notna(avg_r2) and avg_r2 > best_avg_r2:
                 best_avg_r2 = avg_r2
                 best_param_set_avg = param_name
        comparison['best_parameter_set_avg_r2'] = best_param_set_avg
        comparison['best_avg_r2'] = best_avg_r2 if best_param_set_avg is not None else float('nan')


    # Save comparison summary
    comparison_path = os.path.join(base_output_dir, f"{target}_ind_{target_index}_comparison.json")
    with open(comparison_path, 'w') as f:
        # Convert numpy types before saving
        def convert_numpy_types(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32,
                                  np.float64)):
                return float(obj) if pd.notna(obj) else None
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            return obj

        comparison_serializable = convert_numpy_types(comparison)
        json.dump(comparison_serializable, f, indent=4)

    # Generate comparison plot
    if comparison['metrics']:
        param_names = list(comparison['metrics'].keys())
        overall_r2_values = [comparison['metrics'][p].get('overall_r2', np.nan) for p in param_names]
        avg_r2_values = [comparison['metrics'][p].get('avg_r2', np.nan) for p in param_names] # Added avg R2

        plt.figure(figsize=(12, 6))

        # Plot Overall R²
        plt.subplot(1, 2, 1)
        bars1 = plt.bar(param_names, overall_r2_values)
        plt.xlabel('Parameter Set')
        plt.ylabel('Overall R²')
        plt.title(f'Overall R² Comparison ({target} ind {target_index}, {phase})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        # Add value labels
        for bar in bars1:
             height = bar.get_height()
             if pd.notna(height):
                 plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.4f}', ha='center', va='bottom', fontsize=8)

        # Plot Average R²
        plt.subplot(1, 2, 2)
        bars2 = plt.bar(param_names, avg_r2_values)
        plt.xlabel('Parameter Set')
        plt.ylabel('Average R² (across rounds)')
        plt.title(f'Average R² Comparison ({target} ind {target_index}, {phase})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
         # Add value labels
        for bar in bars2:
             height = bar.get_height()
             if pd.notna(height):
                 plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.4f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(base_output_dir, f"{target}_ind_{target_index}_comparison_plot.png"))
        plt.close()

    print(f"\nComparison of parameter sets saved to: {comparison_path}")

    if 'best_parameter_set' in comparison and comparison['best_parameter_set'] is not None:
        print(f"Best parameter set (Overall R²): {comparison['best_parameter_set']} with Overall R² = {comparison['best_overall_r2']:.4f}")
    else:
        print("Could not determine best parameter set based on Overall R².")

    if 'best_parameter_set_avg_r2' in comparison and comparison['best_parameter_set_avg_r2'] is not None:
        print(f"Best parameter set (Average R²): {comparison['best_parameter_set_avg_r2']} with Average R² = {comparison['best_avg_r2']:.4f}")
    else:
         print("Could not determine best parameter set based on Average R².")

    return optimization_results

def main():
    parser = argparse.ArgumentParser(description='Run cross-validation with overall R² calculation and optional time features')

    
    parser.add_argument('--param-set-name', type=str, default=None,
                       help=f'Name of a predefined parameter set to use. Available: {list(PREDEFINED_PARAMETER_SETS.keys())}. If provided, overrides individual hyperparameter args.')
    

    parser.add_argument('--targets', type=str, default=None,
                       help='Target to process, comma-separated (default: show available targets)')

    parser.add_argument('--start', type=int, default=0,
                       help='First CV round to process (default: 0)')

    parser.add_argument('--end', type=int, default=None,
                       help='Last CV round to process (default: all available)')

    parser.add_argument('--step', type=int, default=1,
                       help='Step size for processing rounds (default: 1)')

    parser.add_argument('--output', type=str, default=None,
                       help='Base directory to save results (default: ./results/xgboost_timefeat... or ./results/xgboost_timefeat_optimize...)')

    parser.add_argument('--organized-dir', type=str, default='./src/data/csv',
                       help='Base directory for organized files (default: ./src/data/csv)')

    
    parser.add_argument('--n_estimators', type=int, default=500,
                       help='Number of estimators for XGBoost (default: 500). Ignored if --param-set-name is used.')

    parser.add_argument('--learning_rate', type=float, default=0.05,
                       help='Learning rate for XGBoost (default: 0.05). Ignored if --param-set-name is used.')

    parser.add_argument('--max_depth', type=int, default=6,
                       help='Max depth for XGBoost (default: 6). Ignored if --param-set-name is used.')

    parser.add_argument('--subsample', type=float, default=0.8,
                       help='Subsample ratio for XGBoost (default: 0.8). Ignored if --param-set-name is used.')

    parser.add_argument('--objective', type=str, default='reg:squarederror',
                        help='Objective function for XGBoost (default: reg:squarederror). Ignored if --param-set-name is used.')

    parser.add_argument('--colsample_bytree', type=float, default=0.9,
                        help='Colsample by tree for XGBoost (default: 0.9). Ignored if --param-set-name is used.')

    parser.add_argument('--min_child_weight', type=int, default=1,
                        help='Min child weight for XGBoost (default: 1). Ignored if --param-set-name is used.')

    parser.add_argument('--gamma', type=float, default=0,
                        help='Gamma for XGBoost (default: 0). Ignored if --param-set-name is used.')
    

    parser.add_argument('--list', action='store_true',
                       help='List available targets and exit')

    parser.add_argument('--target-index', type=int, default=None,
                        help='Index of target to process (0 or 1, default: run both)')

    parser.add_argument('--optimize', action='store_true',
                        help='Try multiple predefined parameter sets and compare results (will use time features)')

    parser.add_argument('--phase', type=str, default='validation', choices=['validation', 'test'],
                        help='Phase to use for cross-validation (validation or test) (default: validation)')

    args = parser.parse_args()

    # Get available targets for the specified phase
    available_targets = get_available_targets(args.organized_dir, phase=args.phase)

    if args.list or not available_targets:
        print(f"\nAvailable targets for phase '{args.phase}':")
        for target in available_targets:
            target_dir = os.path.join(args.organized_dir, target)
            try:
                num_rounds = get_cv_round_count(target_dir, phase=args.phase)
                print(f"  {target} ({num_rounds} rounds)")
            except Exception as e:
                print(f"  {target} (Error: {str(e)})")
        return

    custom_model_params = None
    if not args.optimize and not args.param_set_name:
         custom_model_params = {
             'objective': args.objective,
             'n_estimators': args.n_estimators,
             'learning_rate': args.learning_rate,
             'max_depth': args.max_depth,
             'subsample': args.subsample,
             'colsample_bytree': args.colsample_bytree,
             'min_child_weight': args.min_child_weight,
             'gamma': args.gamma,
             'random_state': 42,
             'tree_method': 'hist'
         }


    # Determine targets to process
    target_list = available_targets if args.targets is None else args.targets.split(',')

    # Process each target
    for target in target_list:
        if target not in available_targets:
            print(f"Target '{target}' not found in available targets for phase '{args.phase}'.")
            continue

        try:
            print(f"\n{'='*50}")
            print(f"Processing target: {target} (Phase: {args.phase})")
            print(f"{'='*50}")

            # Determine target indices to process
            if args.target_index is not None:
                target_indices = [args.target_index]
            else:
                target_indices = [0, 1]  # Process both up and down regulation

            for target_index in target_indices:
                if args.optimize:
                    # Try multiple predefined parameter sets
                    try_parameter_sets_for_target(
                        target=target,
                        start_round=args.start,
                        end_round=args.end,
                        step=args.step,
                        organized_dir=args.organized_dir,
                        target_index=target_index,
                        phase=args.phase
                    )
                else:
                    
                    # Run cross-validation with specified parameters (either name or custom)
                    run_cv_for_target(
                        target=target,
                        start_round=args.start,
                        end_round=args.end,
                        step=args.step,
                        model_params=custom_model_params, # Pass custom params if name not used
                        output_dir=args.output,
                        organized_dir=args.organized_dir,
                        target_index=target_index,
                        phase=args.phase,
                        param_set_name=args.param_set_name # Pass the name if provided
                    )
                    
        except Exception as e:
            print(f"Critical error processing target {target}: {type(e).__name__} - {str(e)}")
            # Optionally add traceback here for debugging
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    main()