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
import joblib
from datetime import datetime

# Suppress warnings to reduce output clutter
warnings.filterwarnings('ignore')


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
            # Assume index needs conversion if not already DatetimeIndex
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


def process_single_round_stacked_with_saved_model(round_num, target, target_dir, x_files_dir,
                                                  target_index, ebm_params, xgb_params,
                                                  models_dir=None, best_ebm_param_name=None, phase="validation"):
    """
    Process a single CV round for full stacked model using saved EBM models.
    Loads the saved EBM model if available, otherwise trains a new one.
    Data loading now includes adding time features and phase selection.

    Returns:
        Dictionary with round results or None if error
    """
    try:
        print(f"\nProcessing round {round_num} ({phase}) with stacked model using saved EBM...")

        # Load data for this round (includes adding time features and phase selection)
        X_train, y_train, X_test, y_test = load_cv_round(
            cv_round=round_num,
            target_dir=target_dir,
            x_files_dir=x_files_dir,
            target_index=target_index,
            phase=phase # Pass phase
        )

        # Try to load the saved EBM model if available
        ebm_model = None
        if models_dir is not None and best_ebm_param_name is not None:
            # Include phase in model filename
            model_filename = f"{target}_ind_{target_index}_{best_ebm_param_name}_round_{round_num}_{phase}.joblib"
            model_path = os.path.join(models_dir, model_filename)

            if os.path.exists(model_path):
                print(f"  Loading saved EBM model from {model_path}")
                try:
                    ebm_model = load(model_path)
                except Exception as e:
                    print(f"  Error loading model from {model_path}: {str(e)}")

        # If loading failed or no saved model, train a new one
        if ebm_model is None:
            print(f"  No saved model found or loading failed. Training new EBM model for round {round_num} ({phase})...")
            ebm_model = ExplainableBoostingRegressor(**ebm_params)

            # Handle whether y_train is Series or DataFrame
            if isinstance(y_train, pd.DataFrame):
                y_train_vals = y_train.iloc[:, 0].values
            else: # It's a Series
                y_train_vals = y_train.values

            # Fit EBM model
            ebm_model.fit(X_train, y_train_vals)

        # Train and evaluate stacked model (uses X_train/X_test with time features)
        result = train_and_evaluate_stacked(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            ebm_model=ebm_model,
            xgb_params=xgb_params,
            target=target,
            target_index=target_index
        )

        # Check if we skipped this round due to errors
        if result['xgb_model'] is None:
            print(f"  Skipping round {round_num} ({phase}) due to model training errors.")
            return None

        # Add round number to predictions
        result['predictions']['round'] = round_num

        # Print round results
        print(f"  Round {round_num} ({phase}) - Stacked Model: MAE: {result['metrics']['mae']:.4f}, R²: {result['metrics']['r2']:.4f}")
        print(f"  Round {round_num} ({phase}) - EBM Only: MAE: {result['metrics']['ebm_mae']:.4f}, R²: {result['metrics']['ebm_r2']:.4f}")
        print(f"  Round {round_num} ({phase}) - Improvement: MAE: {result['metrics']['mae_improvement']:.2f}%, R²: {result['metrics']['r2_improvement']:.2f}%")

        return {
            'round_num': round_num,
            'metrics': result['metrics'],
            'predictions': result['predictions'],
            'ebm_model': result['ebm_model'],
            'xgb_model': result['xgb_model']
        }
    except Exception as e:
        print(f"Error processing round {round_num} ({phase}) with stacked model: {str(e)}")
        return None

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

def get_available_targets(organized_dir="./src/data/csv", areas=["no1", "no2", "no3", "no4", "no5"], phase="validation"):
    """
    Get list of available targets from the organized directory structure for a specific phase.

    Args:
        organized_dir: Base directory for organized files
        areas: List of area directories to check
        phase: Either "validation" or "test" to specify which phase's data to check

    Returns:
        List of available targets (directory names)
    """
    targets = []

    # Look for subdirectories that match the specified areas
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
    adding time features to X data and selecting the correct phase.

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
        #print(f"  Adding time features to X_train and X_test for round {cv_round} ({phase})...") # Less verbose
        X_train = add_time_features(X_train)
        X_test = add_time_features(X_test)
        #print(f"  New X_train columns count: {len(X_train.columns)}") # Less verbose logging
        #print(f"  New X_test columns count: {len(X_test.columns)}")
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
    Get the total number of cross-validation rounds available for a target for a specific phase.

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

def train_and_evaluate_ebm(X_train, y_train, X_test, y_test, ebm_params):
    """
    Train an EBM model and evaluate its performance.
    Assumes X_train/X_test may contain added time features.

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
    else: # It's a Series
        y_train_vals = y_train.values

    if isinstance(y_test, pd.DataFrame):
        y_test_vals = y_test.iloc[:, 0].values
    else: # It's a Series
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

    try:
        # Start timing the model training
        start_time = time.time()

        # Train EBM model
        print(f"  Training EBM model with {X_train_clean.shape[1]} features...")
        ebm_model = ExplainableBoostingRegressor(**ebm_params)
        ebm_model.fit(X_train_clean, y_train_vals_clean)

        # Get EBM predictions on test data
        print(f"  Making EBM predictions on test set with {X_test_clean.shape[1]} features...")
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
            'timestamp': X_test_clean.index, # Use index from cleaned X_test
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

def train_and_evaluate_stacked(X_train, y_train, X_test, y_test, ebm_model, xgb_params, target=None, target_index=None):
    """
    Train the full stacked model (EBM + XGBoost on residuals) and evaluate.
    This assumes EBM model is already trained and X_train/X_test may have time features.

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
        # Ensure EBM predicts on data with the same features it was trained on (including time features)
        ebm_train_preds = ebm_model.predict(X_train)
        ebm_test_preds = ebm_model.predict(X_test)

        # Handle whether y_train/y_test are Series or DataFrame
        if isinstance(y_train, pd.DataFrame):
            y_train_vals = y_train.iloc[:, 0].values
        else: # It's a Series
            y_train_vals = y_train.values

        if isinstance(y_test, pd.DataFrame):
            y_test_vals = y_test.iloc[:, 0].values
        else: # It's a Series
            y_test_vals = y_test.values

        # Calculate residuals for training data
        train_residuals = y_train_vals - ebm_train_preds

        # Train XGBoost on the residuals
        print(f"  Training XGBoost model on residuals with {X_train.shape[1]} features...")
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, train_residuals)
        # Save models (EBM and XGBoost residual) after training
        os.makedirs('models', exist_ok=True)
        # Determine filename suffix based on target_index: up if 0, down if 1
        if target is None:
            raise ValueError("Cannot save models: target is None")
        if target_index not in (0, 1):
            raise ValueError(f"Invalid target_index {target_index}: expected 0 (up) or 1 (down)")
        suffix = 'up' if target_index == 0 else 'down'
        # Save EBM model
        ebm_filename = f"ebm_last_run_{target}_{suffix}.joblib"
        joblib.dump(ebm_model, os.path.join('models', ebm_filename), compress=3)
        # Save stacked (XGBoost) residual model
        xgb_filename = f"stacked_last_run_{target}_{suffix}.joblib"
        joblib.dump(xgb_model, os.path.join('models', xgb_filename), compress=3)

        # Get XGBoost predictions on test data (predicting residuals)
        print(f"  Making XGBoost predictions on test set with {X_test.shape[1]} features...")
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
        r2_improvement = (r2 - ebm_r2) / abs(ebm_r2) * 100 if abs(ebm_r2) > 1e-9 else 0 # Avoid division by zero

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

def process_single_round_ebm(round_num, target, target_dir, x_files_dir, target_index, ebm_params,
                             models_dir=None, param_name=None, phase="validation"):
    """
    Process a single CV round for EBM model - helper function for parallelization.
    Data loading now includes adding time features and phase selection.

    Returns:
        Dictionary with round results or None if error
    """
    try:
        print(f"\nProcessing round {round_num} ({phase}) with EBM...")

        # Load data for this round (includes adding time features and phase selection)
        X_train, y_train, X_test, y_test = load_cv_round(
            cv_round=round_num,
            target_dir=target_dir,
            x_files_dir=x_files_dir,
            target_index=target_index,
            phase=phase # Pass phase
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
            print(f"  Skipping round {round_num} ({phase}) due to model training errors or insufficient valid data.")
            return None

        # Save the model if a models directory and parameter name are provided
        if result['model'] is not None and models_dir is not None and param_name is not None:
            # Create a filename for the saved model, including phase
            model_filename = f"{target}_ind_{target_index}_{param_name}_round_{round_num}_{phase}.joblib"
            model_path = os.path.join(models_dir, model_filename)

            # Save the model
            dump(result['model'], model_path)
            print(f"  Saved EBM model to {model_path}")

            # Add the model path to the result
            result['model_path'] = model_path

        # Add round number to predictions
        result['predictions']['round'] = round_num

        # Print round results
        print(f"  Round {round_num} ({phase}) - MAE: {result['metrics']['mae']:.4f}, RMSE: {result['metrics']['rmse']:.4f}, R²: {result['metrics']['r2']:.4f}")

        return {
            'round_num': round_num,
            'metrics': result['metrics'],
            'predictions': result['predictions'],
            'model': result['model'],
            'model_path': result.get('model_path'),  # Add the model path to the returned dict
            'X_train': X_train, # Return data with time features
            'y_train': y_train,
            'X_test': X_test, # Return data with time features
            'y_test': y_test,
            'clean_data': {
                'X_train_clean': result.get('X_train_clean'),
                'y_train_clean': result.get('y_train_clean'),
                'X_test_clean': result.get('X_test_clean'),
                'y_test_clean': result.get('y_test_clean'),
            }
        }
    except Exception as e:
        print(f"Error processing round {round_num} ({phase}): {str(e)}")
        return None

def process_single_round_stacked(round_num, target, target_dir, x_files_dir, target_index,
                                 ebm_params, xgb_params, phase="validation"):
    """
    Process a single CV round for full stacked model - helper function for parallelization.
    Each round gets its own independently trained EBM and XGBoost models.
    Data loading now includes adding time features and phase selection.

    Returns:
        Dictionary with round results or None if error
    """
    try:
        print(f"\nProcessing round {round_num} ({phase}) with stacked model...")

        # Load data for this round (includes adding time features and phase selection)
        X_train, y_train, X_test, y_test = load_cv_round(
            cv_round=round_num,
            target_dir=target_dir,
            x_files_dir=x_files_dir,
            target_index=target_index,
            phase=phase # Pass phase
        )

        # First train EBM model for this round
        print(f"  Training EBM model for round {round_num} ({phase})...")
        ebm_model = ExplainableBoostingRegressor(**ebm_params)

        # Handle whether y_train is Series or DataFrame
        if isinstance(y_train, pd.DataFrame):
            y_train_vals = y_train.iloc[:, 0].values
        else: # It's a Series
            y_train_vals = y_train.values

        # Fit EBM model
        ebm_model.fit(X_train, y_train_vals)

        # Train and evaluate stacked model (uses X_train/X_test with time features)
        result = train_and_evaluate_stacked(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            ebm_model=ebm_model,
            xgb_params=xgb_params,
            target=target,
            target_index=target_index
        )

        # Check if we skipped this round due to errors
        if result['xgb_model'] is None:
            print(f"  Skipping round {round_num} ({phase}) due to model training errors.")
            return None

        # Add round number to predictions
        result['predictions']['round'] = round_num

        # Print round results
        print(f"  Round {round_num} ({phase}) - Stacked Model: MAE: {result['metrics']['mae']:.4f}, R²: {result['metrics']['r2']:.4f}")
        print(f"  Round {round_num} ({phase}) - EBM Only: MAE: {result['metrics']['ebm_mae']:.4f}, R²: {result['metrics']['ebm_r2']:.4f}")
        print(f"  Round {round_num} ({phase}) - Improvement: MAE: {result['metrics']['mae_improvement']:.2f}%, R²: {result['metrics']['r2_improvement']:.2f}%")

        return {
            'round_num': round_num,
            'metrics': result['metrics'],
            'predictions': result['predictions'],
            'ebm_model': result['ebm_model'],
            'xgb_model': result['xgb_model']
        }
    except Exception as e:
        print(f"Error processing round {round_num} ({phase}) with stacked model: {str(e)}")
        return None

def evaluate_ebm_parameter_sets(target, start_round=0, end_round=None, step=1,
                                ebm_parameter_sets=None, output_dir=None,
                                organized_dir="./src/data/csv", target_index=0,
                                parallel=True, max_workers=None, sample=None, phase="validation"):
    """
    Evaluate different EBM parameter sets and find the best one for a specific phase.
    Uses data with added time features.

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
        phase: Either "validation" or "test" to specify which phase's data to use

    Returns:
        Dictionary with the best EBM parameter set and evaluation results
    """


    # Set up paths
    target_dir = os.path.join(organized_dir, target)
    x_files_dir = os.path.join(organized_dir, target)

    # Check if target directory exists
    if not os.path.isdir(target_dir):
        raise ValueError(f"Target directory not found: {target_dir}")

    # Get timestamp for this evaluation
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Create output directory including phase
    if output_dir is None:
        # Add _timefeat and phase to directory name
        output_dir = f"./results/stacked_optimization_timefeat/{timestamp}_{phase}"

    os.makedirs(output_dir, exist_ok=True)

    # Create a models directory for saving trained EBM models, including phase
    models_dir = os.path.join(output_dir, f"saved_ebm_models_{phase}")
    os.makedirs(models_dir, exist_ok=True)

    # Get total number of rounds for the specified phase
    total_rounds = get_cv_round_count(target_dir, phase=phase)

    # Set default end_round if not provided
    if end_round is None:
        end_round = total_rounds
    else:
        end_round = min(end_round, total_rounds)

    # Get list of rounds to process
    if sample is not None:
        rounds_to_process = sample_rounds_evenly(start_round, end_round, sample)
        print(f"Sampling {len(rounds_to_process)} rounds evenly from {start_round} to {end_round-1} for phase '{phase}'")
    else:
        rounds_to_process = list(range(start_round, end_round, step))

    # Store results for each parameter set
    ebm_evaluation_results = {}

    # Try each EBM parameter set
    for param_name, ebm_params in ebm_parameter_sets.items():
        print(f"\n{'='*80}")
        print(f"Evaluating EBM parameter set: {param_name} (Phase: {phase}, with time features)")
        print(f"Parameters: {ebm_params}")
        print(f"{'='*80}")

        # Create directory for this parameter set, including phase
        param_output_dir = os.path.join(output_dir, f"{target}_ind_{target_index}_{param_name}_{phase}")
        os.makedirs(param_output_dir, exist_ok=True)

        # Save configuration
        config = {
            'target': target,
            'target_index': target_index,
            'parameter_set': param_name,
            'ebm_params': ebm_params,
            'timestamp': timestamp,
            'rounds': rounds_to_process,
            'sampled': sample is not None,
            'time_features_added': True, # Indicate time features were used
            'phase': phase # Save phase
        }

        with open(os.path.join(param_output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # Process rounds
        if parallel and len(rounds_to_process) > 1:
            # Use multiprocessing for parallelism
            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count(), len(rounds_to_process))

            print(f"Running with {max_workers} parallel workers for phase '{phase}'")

            # Create a partial function with fixed arguments
            process_func = partial(
                process_single_round_ebm,
                target=target,
                target_dir=target_dir,
                x_files_dir=x_files_dir,
                target_index=target_index,
                ebm_params=ebm_params,
                models_dir=models_dir,  # Pass models directory
                param_name=param_name,   # Pass parameter set name
                phase=phase # Pass phase
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
                    ebm_params,
                    models_dir=models_dir,  # Pass models directory
                    param_name=param_name,   # Pass parameter set name
                    phase=phase # Pass phase
                )

                if result is not None:
                    results.append(result)

        # Process the results
        if not results:
            print(f"No successful rounds for parameter set {param_name} in phase '{phase}'")
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
        # Add perfect prediction line only if data exists and is valid
        if len(overall_actual) > 0 and pd.notna(overall_actual.min()) and pd.notna(overall_predicted.min()):
             min_val = min(overall_actual.min(), overall_predicted.min())
             max_val = max(overall_actual.max(), overall_predicted.max())
             if pd.notna(min_val) and pd.notna(max_val):
                 plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'EBM {param_name} - Overall Performance ({phase.capitalize()} Phase, R² = {overall_r2:.4f})')
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
            'sampled': sample is not None,
            'time_features_added': True, # Indicate time features were used
            'phase': phase # Save phase
        }

        # Save summary
        with open(os.path.join(param_output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        # Print summary
        print(f"\nSummary for parameter set {param_name} (Phase: {phase}):")
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
        raise ValueError(f"No successful parameter sets found for phase '{phase}'")

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
        'sample_size': len(rounds_to_process) if sample is not None else None,
        'time_features_added': True, # Indicate time features were used
        'phase': phase # Save phase
    }

    for param_name, eval_result in ebm_evaluation_results.items():
        comparison['metrics'][param_name] = {
            'overall_mae': eval_result['summary']['overall_mae'],
            'overall_rmse': eval_result['summary']['overall_rmse'],
            'overall_r2': eval_result['summary']['overall_r2'],
            'avg_training_time': eval_result['summary']['avg_training_time']
        }

    # Save comparison
    with open(os.path.join(output_dir, f"{target}_ind_{target_index}_ebm_comparison_{phase}.json"), 'w') as f:
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
    plt.title(f'Comparison of EBM Parameter Sets for {target}, index {target_index} ({phase.capitalize()} Phase){sampling_info} (with Time Features)')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target}_ind_{target_index}_ebm_comparison_{phase}.png"))
    plt.close()

    print(f"\n{'='*80}")
    print(f"Best EBM parameter set for phase '{phase}': {best_param_set} with R² = {best_r2:.4f}")
    print(f"{'='*80}")

    # Return the best parameter set and all results
    return {
        'best_parameter_set': best_param_set,
        'best_params': ebm_evaluation_results[best_param_set]['params'],
        'best_r2': best_r2,
        'all_results': ebm_evaluation_results,
        'comparison': comparison,
        'output_dir': output_dir,
        'models_dir': models_dir,  # Return the models directory path
        'sampled': sample is not None,
        'sample_size': len(rounds_to_process) if sample is not None else None
    }

def evaluate_stacked_model(target, best_ebm_params, start_round=0, end_round=None, step=1,
                           xgb_parameter_sets=None, output_dir=None, models_dir=None,
                           best_ebm_param_name=None, organized_dir="./src/data/csv", target_index=0,
                           parallel=True, max_workers=None, sample=None, phase="validation"):
    """
    Evaluate different XGBoost parameter sets for the residuals model using saved EBM models for a specific phase.
    Uses data with added time features.

    Args:
        target: Target directory name
        best_ebm_params: Parameters for the EBM model
        start_round: First CV round to process
        end_round: Last CV round to process (None = all available)
        step: Step size for processing rounds
        xgb_parameter_sets: Dictionary of parameter sets to try for XGBoost
        output_dir: Directory to save results
        models_dir: Directory containing saved EBM models
        best_ebm_param_name: Name of the best EBM parameter set (used for loading models)
        organized_dir: Base directory for organized files
        target_index: Index of target variable (0 or 1) for regulation up/down
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers (None = auto)
        sample: If specified, sample this many rounds evenly instead of using all rounds
        phase: Either "validation" or "test" to specify which phase's data to use

    Returns:
        Dictionary with the best stacked model configuration and evaluation results
    """


    # Set up paths
    target_dir = os.path.join(organized_dir, target)
    x_files_dir = os.path.join(organized_dir, target)

    # Check if target directory exists
    if not os.path.isdir(target_dir):
        raise ValueError(f"Target directory not found: {target_dir}")

    # Get timestamp for this evaluation if output_dir not provided
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Add _timefeat and phase to directory name
        output_dir = f"./results/stacked_optimization_timefeat/{timestamp}_{phase}"

    # Create subdirectory for XGBoost evaluation, including phase
    xgb_output_dir = os.path.join(output_dir, f'xgboost_eval_{phase}')
    os.makedirs(xgb_output_dir, exist_ok=True)

    # Get total number of rounds for the specified phase
    total_rounds = get_cv_round_count(target_dir, phase=phase)

    # Set default end_round if not provided
    if end_round is None:
        end_round = total_rounds
    else:
        end_round = min(end_round, total_rounds)

    # Get list of rounds to process
    if sample is not None:
        rounds_to_process = sample_rounds_evenly(start_round, end_round, sample)
        print(f"Sampling {len(rounds_to_process)} rounds evenly from {start_round} to {end_round-1} for phase '{phase}'")
    else:
        rounds_to_process = list(range(start_round, end_round, step))

    print(f"\n{'='*80}")
    print(f"Evaluating stacked models for target: {target}, index: {target_index} (Phase: {phase}, with time features)")
    print(f"Using EBM parameters: {best_ebm_params}")
    print(f"Using saved EBM models when available to speed up optimization")
    print(f"{'='*80}")

    # Store results for each XGBoost parameter set
    xgb_evaluation_results = {}

    # Try each XGBoost parameter set
    for param_name, xgb_params in xgb_parameter_sets.items():
        print(f"\n{'='*80}")
        print(f"Evaluating XGBoost parameter set: {param_name} (Phase: {phase})")
        print(f"Parameters: {xgb_params}")
        print(f"{'='*80}")

        # Create directory for this parameter set, including phase
        param_output_dir = os.path.join(xgb_output_dir, f"{target}_ind_{target_index}_{param_name}_{phase}")
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
            'sample_size': len(rounds_to_process) if sample is not None else None,
            'time_features_added': True, # Indicate time features were used
            'phase': phase # Save phase
        }

        with open(os.path.join(param_output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # Process rounds
        if parallel and len(rounds_to_process) > 1:
            # Use multiprocessing for parallelism
            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count(), len(rounds_to_process))

            print(f"Running with {max_workers} parallel workers for phase '{phase}'")

            # Create a partial function with fixed arguments
            process_func = partial(
                process_single_round_stacked_with_saved_model,  # Use new function that loads models
                target=target,
                target_dir=target_dir,
                x_files_dir=x_files_dir,
                target_index=target_index,
                ebm_params=best_ebm_params,
                xgb_params=xgb_params,
                models_dir=models_dir,
                best_ebm_param_name=best_ebm_param_name,
                phase=phase # Pass phase
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
                result = process_single_round_stacked_with_saved_model(
                    round_num,
                    target,
                    target_dir,
                    x_files_dir,
                    target_index,
                    best_ebm_params,
                    xgb_params,
                    models_dir=models_dir,
                    best_ebm_param_name=best_ebm_param_name,
                    phase=phase # Pass phase
                )

                if result is not None:
                    results.append(result)

        # Process the results
        if not results:
            print(f"No successful rounds for parameter set {param_name} in phase '{phase}'")
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

        overall_mae_improvement = (overall_ebm_mae - overall_mae) / overall_ebm_mae * 100 if overall_ebm_mae > 0 else 0
        overall_rmse_improvement = (overall_ebm_rmse - overall_rmse) / overall_ebm_rmse * 100 if overall_ebm_rmse > 0 else 0
        overall_r2_improvement = (overall_r2 - overall_ebm_r2) / abs(overall_ebm_r2) * 100 if abs(overall_ebm_r2) > 1e-9 else 0

        # Save combined predictions
        all_predictions.to_csv(os.path.join(param_output_dir, 'all_predictions.csv'), index=False)

        # Save combined evaluation plot
        plt.figure(figsize=(12, 8))

        # Plot 1: Stacked Model Performance
        plt.subplot(2, 2, 1)
        plt.scatter(overall_actual, overall_predicted, alpha=0.5)
        # Add perfect prediction line only if data exists and is valid
        if len(overall_actual) > 0 and pd.notna(overall_actual.min()) and pd.notna(overall_predicted.min()):
             min_val = min(overall_actual.min(), overall_predicted.min())
             max_val = max(overall_actual.max(), overall_predicted.max())
             if pd.notna(min_val) and pd.notna(max_val):
                 plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Stacked Model Performance ({phase.capitalize()} Phase, R² = {overall_r2:.4f})')
        plt.grid(True)

        # Plot 2: EBM Only Performance
        plt.subplot(2, 2, 2)
        plt.scatter(overall_actual, overall_ebm_pred, alpha=0.5)
        # Add perfect prediction line only if data exists and is valid
        if len(overall_actual) > 0 and pd.notna(overall_actual.min()) and pd.notna(overall_ebm_pred.min()):
             min_val = min(overall_actual.min(), overall_ebm_pred.min())
             max_val = max(overall_actual.max(), overall_ebm_pred.max())
             if pd.notna(min_val) and pd.notna(max_val):
                 plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'EBM Only Performance ({phase.capitalize()} Phase, R² = {overall_ebm_r2:.4f})')
        plt.grid(True)

        # Plot 3: Improvement in R² Across Rounds
        plt.subplot(2, 2, 3)
        r2_improvements = [r['metrics']['r2_improvement'] for r in results]
        round_nums = [r['round_num'] for r in results]
        plt.bar(round_nums, r2_improvements)
        plt.axhline(y=overall_r2_improvement, color='r', linestyle='-', label=f'Avg: {overall_r2_improvement:.2f}%')
        plt.xlabel('Round Number')
        plt.ylabel('R² Improvement (%)')
        plt.title(f'R² Improvement by Round ({phase.capitalize()} Phase)')
        plt.legend()
        plt.grid(True, axis='y')

        # Plot 4: Improvement in MAE Across Rounds
        plt.subplot(2, 2, 4)
        mae_improvements = [r['metrics']['mae_improvement'] for r in results]
        plt.bar(round_nums, mae_improvements)
        plt.axhline(y=overall_mae_improvement, color='r', linestyle='-', label=f'Avg: {overall_mae_improvement:.2f}%')
        plt.xlabel('Round Number')
        plt.ylabel('MAE Improvement (%)')
        plt.title(f'MAE Improvement by Round ({phase.capitalize()} Phase)')
        plt.legend()
        plt.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(param_output_dir, 'stacked_model_performance.png'))
        plt.close()

        # We won't save a single model since each round has its own model
        # But we can save the parameters
        model_params = {
            'ebm_params': best_ebm_params,
            'xgb_params': xgb_params
        }

        with open(os.path.join(param_output_dir, 'model_params.json'), 'w') as f:
            json.dump(model_params, f, indent=4)

        # Summarize results
        summary = {
            'ebm_parameter_set': best_ebm_param_name,
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
            'sample_size': len(rounds_to_process) if sample is not None else None,
            'time_features_added': True, # Indicate time features were used
            'phase': phase # Save phase
        }

        # Save summary
        with open(os.path.join(param_output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        # Print summary
        print(f"\nSummary for XGBoost parameter set {param_name} (Phase: {phase}):")
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
        raise ValueError(f"No successful XGBoost parameter sets found for phase '{phase}'")

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
        'sample_size': len(rounds_to_process) if sample is not None else None,
        'time_features_added': True, # Indicate time features were used
        'phase': phase # Save phase
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
    with open(os.path.join(xgb_output_dir, f"{target}_ind_{target_index}_xgb_comparison_{phase}.json"), 'w') as f:
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
    plt.title(f'Comparison of XGBoost Parameter Sets for {target}, index {target_index} ({phase.capitalize()} Phase){sampling_info} (with Time Features)')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom')

    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(xgb_output_dir, f"{target}_ind_{target_index}_xgb_comparison_{phase}.png"))
    plt.close()

    print(f"\n{'='*80}")
    print(f"Best XGBoost parameter set for phase '{phase}': {best_param_set} with R² improvement = {best_r2_improvement:.2f}%")
    print(f"{'='*80}")

    # After finding the best XGBoost parameter set
    # Delete all EBM models except the ones used by the best parameter set
    if models_dir is not None and best_ebm_param_name is not None:
        print(f"\nCleaning up saved EBM models for phase '{phase}', keeping only the best ones...")
        model_files = os.listdir(models_dir)
        # Include phase in the pattern
        best_model_pattern = f"{target}_ind_{target_index}_{best_ebm_param_name}_round_"

        for model_file in model_files:
            # Check if the file belongs to the current phase and target/index
            if model_file.startswith(f"{target}_ind_{target_index}_") and model_file.endswith(f"_{phase}.joblib"):
                # Keep only if it matches the best EBM param name
                if not model_file.startswith(best_model_pattern):
                    model_path = os.path.join(models_dir, model_file)
                    try:
                        os.remove(model_path)
                        print(f"  Deleted {model_file}")
                    except Exception as e:
                        print(f"  Error deleting {model_file}: {str(e)}")

    # Save the best model parameters (not actual models since each round has its own)
    best_model_params = {
        'ebm_params': best_ebm_params,
        'xgb_params': xgb_evaluation_results[best_param_set]['xgb_params'],
        'target': target,
        'target_index': target_index,
        'time_features_added': True, # Indicate time features were used
        'phase': phase # Save phase
    }

    with open(os.path.join(xgb_output_dir, f"{target}_ind_{target_index}_best_model_params_{phase}.json"), 'w') as f:
        json.dump(best_model_params, f, indent=4)

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
        'sample_size': len(rounds_to_process) if sample is not None else None,
        'time_features_added': True, # Indicate time features were used
        'phase': phase # Save phase
    }

    # Save final summary
    with open(os.path.join(output_dir, f"{target}_ind_{target_index}_final_summary_{phase}.json"), 'w') as f:
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
    plt.title(f'Performance Comparison for {target}, index {target_index} ({phase.capitalize()} Phase){sampling_info} (with Time Features)')

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
    plt.savefig(os.path.join(output_dir, f"{target}_ind_{target_index}_final_comparison_{phase}.png"))
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
                           parallel=True, max_workers=None, sample=None,
                           only_optimize_xgb=False, optimal_ebm_param_name=None, phase="validation"):
    """
    Optimize a stacked EBM+XGBoost model for a target using data with time features for a specific phase.
    First finds the best EBM model, then the best XGBoost model for the residuals.
    Uses model saving to speed up optimization.

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
        only_optimize_xgb: If True, skip EBM optimization and use provided optimal_ebm_param_name
        optimal_ebm_param_name: Name of the optimal EBM parameter set to use if only_optimize_xgb is True
        phase: Either "validation" or "test" to specify which phase's data to use

    Returns:
        Dictionary with the optimized model configuration and evaluation results
    """
    # Create timestamp for this optimization run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Add _timefeat and phase to directory name
    output_dir = f"./results/stacked_optimization_timefeat/{timestamp}_{phase}"

    # Step 1: Find the best EBM model
    print(f"\n{'='*80}")
    print(f"Step 1: Finding the best EBM model for target: {target}, index: {target_index} (Phase: {phase}, with time features)")
    sampling_info = f" (Sampling {sample} rounds)" if sample is not None else ""
    print(f"Rounds: {start_round} to {end_round if end_round is not None else 'all'}{sampling_info}")
    print(f"{'='*80}")

    if not only_optimize_xgb:
        ebm_results = evaluate_ebm_parameter_sets(
            target=target,
            start_round=start_round,
            end_round=end_round,
            step=step,
            ebm_parameter_sets=ebm_parameter_sets,
            output_dir=output_dir, # Pass the main output dir
            organized_dir=organized_dir,
            target_index=target_index,
            parallel=parallel,
            max_workers=max_workers,
            sample=sample,
            phase=phase # Pass phase
        )
        best_ebm_name = ebm_results['best_parameter_set']
        best_ebm_params = ebm_results['best_params']
        models_dir = ebm_results['models_dir']
    else:
        # Use the provided optimal EBM parameters directly
        if optimal_ebm_param_name is None or optimal_ebm_param_name not in ebm_parameter_sets:
            raise ValueError(f"Invalid optimal_ebm_param_name: {optimal_ebm_param_name}. Available options: {list(ebm_parameter_sets.keys())}")

        best_ebm_name = optimal_ebm_param_name
        best_ebm_params = ebm_parameter_sets[best_ebm_name]
        print(f"Using specified optimal EBM parameter set: {best_ebm_name} for phase '{phase}'")

        # Create a models directory if one doesn't exist yet, including phase
        models_dir = os.path.join(output_dir, f"saved_ebm_models_{phase}")
        os.makedirs(models_dir, exist_ok=True)

    # Step 2: Find the best XGBoost model for the residuals
    print(f"\n{'='*80}")
    print(f"Step 2: Finding the best XGBoost model for the residuals (Phase: {phase}, with time features)")
    print(f"Using best EBM model parameters: {best_ebm_name}")
    print(f"{'='*80}")

    stacked_results = evaluate_stacked_model(
        target=target,
        best_ebm_params=best_ebm_params,
        start_round=start_round,
        end_round=end_round,
        step=step,
        xgb_parameter_sets=xgb_parameter_sets,
        output_dir=output_dir, # Pass the main output dir
        models_dir=models_dir,  # Pass the models directory
        best_ebm_param_name=best_ebm_name,  # Pass the name of best param set
        organized_dir=organized_dir,
        target_index=target_index,
        parallel=parallel,
        max_workers=max_workers,
        sample=sample,
        phase=phase # Pass phase
    )

    # Combine results
    optimization_results = {
        'target': target,
        'target_index': target_index,
        'best_ebm_parameter_set': best_ebm_name,
        'best_ebm_params': best_ebm_params,
        'best_xgb_params': stacked_results['best_xgb_params'],
        'ebm_only_r2': stacked_results['ebm_only_r2'],
        'stacked_r2': stacked_results['stacked_r2'],
        'r2_improvement': stacked_results['r2_improvement'],
        'output_dir': output_dir,
        'timestamp': timestamp,
        'sampled': sample is not None,
        'sample_size': stacked_results.get('sample_size'),
        'time_features_added': True, # Indicate time features were used
        'phase': phase # Save phase
    }

    # Save the final optimization results
    with open(os.path.join(output_dir, f"{target}_ind_{target_index}_optimization_results_{phase}.json"), 'w') as f:
        json.dump(optimization_results, f, indent=4)

    print(f"\n{'='*80}")
    print(f"Optimization complete for target: {target}, index: {target_index} (Phase: {phase}, with time features)")
    print(f"Best EBM model: {best_ebm_name}")
    print(f"EBM Only R²: {stacked_results['ebm_only_r2']:.4f}")
    print(f"Stacked Model R²: {stacked_results['stacked_r2']:.4f}")
    print(f"R² Improvement: {stacked_results['r2_improvement']:.2f}%")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")

    return optimization_results

def run_stacked_model_with_params(target, ebm_params, xgb_params, start_round=0, end_round=None, step=1,
                                  output_dir=None, organized_dir="./src/data/csv", target_index=0,
                                  parallel=True, max_workers=None, sample=None, phase="validation"):
    """
    Run the stacked model with specified parameters on all rounds using data with time features for a specific phase.
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
        phase: Either "validation" or "test" to specify which phase's data to use

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

    # Determine where to save results for this target/index
    if output_dir is None:
        # Build default base directory with timestamp and phase
        base_output_dir = f"./results/stacked_timefeat/{timestamp}_{phase}"
        os.makedirs(base_output_dir, exist_ok=True)
        target_output_dir = os.path.join(base_output_dir, f"{target}_ind_{target_index}_{phase}")
        os.makedirs(target_output_dir, exist_ok=True)
    else:
        # Use provided output_dir as the target output folder
        target_output_dir = output_dir
        os.makedirs(target_output_dir, exist_ok=True)

    # Get total number of rounds for the specified phase
    total_rounds = get_cv_round_count(target_dir, phase=phase)

    # Set default end_round if not provided
    if end_round is None:
        end_round = total_rounds
    else:
        end_round = min(end_round, total_rounds)

    # Get list of rounds to process
    if sample is not None:
        rounds_to_process = sample_rounds_evenly(start_round, end_round, sample)
        print(f"Sampling {len(rounds_to_process)} rounds evenly from {start_round} to {end_round-1} for phase '{phase}'")
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
        'sample_size': len(rounds_to_process) if sample is not None else None,
        'time_features_added': True, # Indicate time features were used
        'phase': phase # Save phase
    }

    with open(os.path.join(target_output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\n{'='*80}")
    print(f"Running stacked model for target: {target}, index: {target_index} (Phase: {phase}, with time features)")
    sampling_info = f" (Sampled {len(rounds_to_process)} rounds)" if sample is not None else ""
    print(f"Rounds: {start_round} to {end_round-1} with step {step}{sampling_info}")
    print(f"Total rounds to process: {len(rounds_to_process)}")
    print(f"{'='*80}")

    # Process each round
    if parallel and len(rounds_to_process) > 1:
        # Use multiprocessing for parallelism
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(rounds_to_process))

        print(f"Running with {max_workers} parallel workers for phase '{phase}'")

        # Create a partial function with fixed arguments
        process_func = partial(
            process_single_round_stacked,
            target=target,
            target_dir=target_dir,
            x_files_dir=x_files_dir,
            target_index=target_index,
            ebm_params=ebm_params,
            xgb_params=xgb_params,
            phase=phase # Pass phase
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
                xgb_params,
                phase=phase # Pass phase
            )

            if result is not None:
                results.append(result)

    # Check if we have any successful rounds
    if not results:
        print(f"No successful rounds for target {target}, index {target_index}, phase {phase}")
        return {
            'error': 'No successful rounds',
            'target': target,
            'target_index': target_index,
            'phase': phase
        }

    # Save individual round results
    for result in results:
        round_num = result['round_num']

        # Save predictions
        result['predictions'].to_csv(os.path.join(target_output_dir, f"round_{round_num}_predictions.csv"), index=False)

        # Create and save plot
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        # Ensure timestamps are sorted for plotting if needed
        pred_df_sorted = result['predictions'].sort_values('timestamp')
        plt.plot(pred_df_sorted['timestamp'], pred_df_sorted['actual'], 'b-', label='Actual', alpha=0.7)
        plt.plot(pred_df_sorted['timestamp'], pred_df_sorted['predicted'], 'r--', label='Stacked Model', alpha=0.7)
        plt.plot(pred_df_sorted['timestamp'], pred_df_sorted['ebm_pred'], 'g-.', label='EBM Only', alpha=0.5)
        plt.title(f'Round {round_num} ({phase.capitalize()}) - Actual vs Predictions')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(result['predictions']['actual'], result['predictions']['predicted'], alpha=0.5, label='Stacked')
        plt.scatter(result['predictions']['actual'], result['predictions']['ebm_pred'], alpha=0.3, label='EBM Only')
        # Add perfect prediction line only if data exists and is valid
        if not result['predictions'].empty:
            actual_vals = result['predictions']['actual']
            predicted_vals = result['predictions']['predicted']
            ebm_vals = result['predictions']['ebm_pred']
            min_val = min(actual_vals.min(), predicted_vals.min(), ebm_vals.min())
            max_val = max(actual_vals.max(), predicted_vals.max(), ebm_vals.max())
            if pd.notna(min_val) and pd.notna(max_val):
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Round {round_num} ({phase.capitalize()}) - Correlation')
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

    overall_mae_improvement = (overall_ebm_mae - overall_mae) / overall_ebm_mae * 100 if overall_ebm_mae > 0 else 0
    overall_rmse_improvement = (overall_ebm_rmse - overall_rmse) / overall_ebm_rmse * 100 if overall_ebm_rmse > 0 else 0
    overall_r2_improvement = (overall_r2 - overall_ebm_r2) / abs(overall_ebm_r2) * 100 if abs(overall_ebm_r2) > 1e-9 else 0

    # Save combined predictions
    all_predictions.to_csv(os.path.join(target_output_dir, 'all_predictions.csv'), index=False)

    # Create overall evaluation plots
    plt.figure(figsize=(15, 10))

    # Plot 1: Stacked Model Performance
    plt.subplot(2, 2, 1)
    plt.scatter(overall_actual, overall_predicted, alpha=0.5)
    # Add perfect prediction line only if data exists and is valid
    if len(overall_actual) > 0 and pd.notna(overall_actual.min()) and pd.notna(overall_predicted.min()):
        min_val = min(overall_actual.min(), overall_predicted.min())
        max_val = max(overall_actual.max(), overall_predicted.max())
        if pd.notna(min_val) and pd.notna(max_val):
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    sampling_info = f" (Sampled {len(rounds_to_process)} rounds)" if sample is not None else ""
    plt.title(f'Stacked Model Performance ({phase.capitalize()} Phase, R² = {overall_r2:.4f}){sampling_info} (with Time Features)')
    plt.grid(True)

    # Plot 2: EBM Only Performance
    plt.subplot(2, 2, 2)
    plt.scatter(overall_actual, overall_ebm_pred, alpha=0.5)
    # Add perfect prediction line only if data exists and is valid
    if len(overall_actual) > 0 and pd.notna(overall_actual.min()) and pd.notna(overall_ebm_pred.min()):
        min_val = min(overall_actual.min(), overall_ebm_pred.min())
        max_val = max(overall_actual.max(), overall_ebm_pred.max())
        if pd.notna(min_val) and pd.notna(max_val):
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'EBM Only Performance ({phase.capitalize()} Phase, R² = {overall_ebm_r2:.4f})')
    plt.grid(True)

    # Plot 3: Improvement in R² Across Rounds
    plt.subplot(2, 2, 3)
    r2_improvements = [r['metrics']['r2_improvement'] for r in results]
    round_nums = [r['round_num'] for r in results]
    plt.bar(round_nums, r2_improvements)
    plt.axhline(y=overall_r2_improvement, color='r', linestyle='-', label=f'Avg: {overall_r2_improvement:.2f}%')
    plt.xlabel('Round Number')
    plt.ylabel('R² Improvement (%)')
    plt.title(f'R² Improvement by Round ({phase.capitalize()} Phase)')
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
    plt.title(f'Performance Comparison ({phase.capitalize()} Phase)')

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
    plt.title(f'MAE by Round ({phase.capitalize()} Phase)')
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
    plt.title(f'R² by Round ({phase.capitalize()} Phase)')
    plt.legend()
    plt.grid(True, axis='y')

    # Plot 3: MAE Improvement by Round
    plt.subplot(2, 2, 3)
    plt.bar(round_nums, metrics['mae_improvement'])
    plt.axhline(y=overall_mae_improvement, color='r', linestyle='-', label=f'Avg: {overall_mae_improvement:.2f}%')
    plt.xlabel('Round Number')
    plt.ylabel('MAE Improvement (%)')
    plt.title(f'MAE Improvement by Round ({phase.capitalize()} Phase)')
    plt.legend()
    plt.grid(True, axis='y')

    # Plot 4: R² Improvement by Round
    plt.subplot(2, 2, 4)
    plt.bar(round_nums, metrics['r2_improvement'])
    plt.axhline(y=overall_r2_improvement, color='r', linestyle='-', label=f'Avg: {overall_r2_improvement:.2f}%')
    plt.xlabel('Round Number')
    plt.ylabel('R² Improvement (%)')
    plt.title(f'R² Improvement by Round ({phase.capitalize()} Phase)')
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
        'sample_size': len(rounds_to_process) if sample is not None else None,
        'time_features_added': True, # Indicate time features were used
        'phase': phase # Save phase
    }

    # Save summary
    with open(os.path.join(target_output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # Save model parameters (not the actual models)
    model_params = {
        'ebm_params': ebm_params,
        'xgb_params': xgb_params,
        'target': target,
        'target_index': target_index,
        'time_features_added': True, # Indicate time features were used
        'phase': phase # Save phase
    }

    with open(os.path.join(target_output_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=4)

    # Print summary
    print(f"\nStacked Model Results Summary (Phase: {phase}, with time features):")
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
    parser = argparse.ArgumentParser(description='Optimize and run stacked EBM+XGBoost models with time features')

    parser.add_argument('--targets', type=str, default=None,
                        help='Target to process, comma-separated (default: show available targets)')

    parser.add_argument('--start', type=int, default=0,
                        help='First CV round to process (default: 0)')

    parser.add_argument('--end', type=int, default=None,
                        help='Last CV round to process (default: all available)')

    parser.add_argument('--step', type=int, default=1,
                        help='Step size for processing rounds (default: 1)')

    parser.add_argument('--output', type=str, default=None,
                        help='Base directory to save results (default: ./results/stacked_timefeat/DATE_PHASE or ./results/stacked_optimization_timefeat/DATE_PHASE)')

    parser.add_argument('--organized-dir', type=str, default='./src/data/csv',
                        help='Base directory for organized files (default: ./src/data/csv)')

    parser.add_argument('--list', action='store_true',
                        help='List available targets and exit')

    parser.add_argument('--target-index', type=int, default=None,
                         help='Index of target to process (0 or 1, default: run both)')

    parser.add_argument('--optimize', action='store_true',
                        help='Run sequential optimization to find the best stacked model (will use time features)')

    parser.add_argument('--run-best', action='store_true',
                        help='Run with the best previously found parameters (use names or read from JSON file, assumes time features were used)')

    
    parser.add_argument('--best-ebm-param-name', type=str, default=None,
                        help='Name of the best EBM parameter set (used with --run-best, alternative to --best-params-file)')
    parser.add_argument('--best-xgb-param-name', type=str, default=None,
                        help='Name of the best XGBoost parameter set (used with --run-best, alternative to --best-params-file)')
    parser.add_argument('--best-params-file', type=str, default=None,
                        help='JSON file with best parameters (used with --run-best if names are not provided)')
    

    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')

    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: auto)')

    parser.add_argument('--sample', type=int, default=None,
                        help='Sample this many rounds evenly (default: use all rounds)')

    parser.add_argument('--optimal_ebm_params', type=str, default=None,
                        help='Name of optimal EBM parameter set to use instead of optimizing (used with --optimize)')

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

    # Define EBM parameter sets
    ebm_parameter_sets = {
            #In theory, EBM should perform reasonably well out of the box
            'ebm_default': {
                'interactions': 0,
                'outer_bags': 8,
                'inner_bags': 0,
                'learning_rate': 0.01,
                'validation_size': 0.15,
                'min_samples_leaf': 2,
                'max_leaves': 256,
                'random_state': 42
            },
            'ebm_fast': {
                'max_bins': 64,
                'max_interaction_bins': 16,
                'interactions': 0,
                'learning_rate': 0.05,
                'min_samples_leaf': 10,
                'random_state': 42,
                'outer_bags': 4,
                'inner_bags': 0,
                'max_rounds': 1000,
                'early_stopping_rounds': 50
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
        'xgb_residual_mse': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.03,
            'n_estimators': 300,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_lambda': 1,  # Added regularization
            'random_state': 42,
            'tree_method': 'hist' # Added for consistency
        },
        'xgb_residual_huber': {
            'objective': 'reg:pseudohubererror',
            'learning_rate': 0.02,
            'n_estimators': 400,
            'max_depth': 4,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'gamma': 0.1,
            'min_child_weight': 2,  # Added for consistency
            'reg_lambda': 1,  # Added regularization
            'random_state': 42,
            'tree_method': 'hist' # Added for consistency
        },
        'xgb_residual_robust': {  # Changed from quantile to robust
            'objective': 'reg:squarederror',
            'learning_rate': 0.01,  # Slower learning rate
            'n_estimators': 500,
            'max_depth': 2,  # Very shallow trees for robustness
            'subsample': 0.6,  # More aggressive subsampling
            'colsample_bytree': 0.6,
            'min_child_weight': 5,  # More conservative
            'reg_lambda': 3,  # Stronger regularization
            'random_state': 42,
            'tree_method': 'hist' # Added for consistency
        }
    }

    # Determine targets to process
    target_list = available_targets if args.targets is None else args.targets.split(',')

    # Process each target
    for target in target_list:
        if target not in available_targets:
            print(f"Target '{target}' not found in available targets for phase '{args.phase}'.")
            continue

        try:
            print(f"\n{'='*80}")
            print(f"Processing target: {target} (Phase: {args.phase}, with time features)")
            print(f"{'='*80}")

            # Determine target indices to process
            if args.target_index is not None:
                target_indices = [args.target_index]
            else:
                target_indices = [0, 1]  # Process both indices

            for target_index in target_indices:
                if args.optimize:
                    if args.optimal_ebm_params is not None:
                        # Use optimal EBM parameters
                        if args.optimal_ebm_params not in ebm_parameter_sets:
                            print(f"Error: EBM parameter set '{args.optimal_ebm_params}' not found")
                            continue

                        print(f"Using optimal EBM parameter set: {args.optimal_ebm_params} for phase '{args.phase}'")

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
                            sample=args.sample,
                            only_optimize_xgb=True,
                            optimal_ebm_param_name=args.optimal_ebm_params,
                            phase=args.phase
                        )
                    else:
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
                            sample=args.sample,
                            phase=args.phase
                        )
                elif args.run_best:
                    
                    ebm_params = None
                    xgb_params = None
                    source_info = ""

                    
                    if args.best_ebm_param_name and args.best_xgb_param_name:
                        if args.best_ebm_param_name in ebm_parameter_sets:
                            ebm_params = ebm_parameter_sets[args.best_ebm_param_name]
                        else:
                            print(f"Error: EBM parameter set name '{args.best_ebm_param_name}' not found in script definitions.")
                            continue # Skip to next target/index

                        if args.best_xgb_param_name in xgb_residual_sets:
                            xgb_params = xgb_residual_sets[args.best_xgb_param_name]
                        else:
                             print(f"Error: XGBoost parameter set name '{args.best_xgb_param_name}' not found in script definitions.")
                             continue # Skip to next target/index

                        if ebm_params and xgb_params:
                             source_info = f"using names '{args.best_ebm_param_name}' and '{args.best_xgb_param_name}'"

                    # Fallback to using the parameters file
                    elif args.best_params_file:
                         source_info = f"from file {args.best_params_file}"
                         try:
                            with open(args.best_params_file, 'r') as f:
                                best_params_all = json.load(f)
                         except FileNotFoundError:
                             print(f"Error: Best parameters file not found: {args.best_params_file}")
                             continue
                         except json.JSONDecodeError:
                             print(f"Error: Could not decode JSON from {args.best_params_file}")
                             continue

                         # Try to find parameters in the loaded JSON (existing logic)
                         target_key_opt = f"{target}_ind_{target_index}_optimization_results_{args.phase}.json"
                         target_key_run = f"{target}_ind_{target_index}_{args.phase}"
                         target_key_final_summary = f"{target}_ind_{target_index}_final_summary_{args.phase}.json"
                         params_found_in_file = False

                         if target_key_opt in best_params_all:
                            params = best_params_all[target_key_opt]
                            ebm_params = params.get('best_ebm_params')
                            xgb_params = params.get('best_xgb_params')
                            params_found_in_file = ebm_params and xgb_params
                         elif target_key_run in best_params_all:
                            params = best_params_all[target_key_run]
                            ebm_params = params.get('ebm_params')
                            xgb_params = params.get('xgb_params')
                            params_found_in_file = ebm_params and xgb_params
                         elif target_key_final_summary in best_params_all:
                             params = best_params_all[target_key_final_summary]
                             ebm_params = params.get('best_ebm_params')
                             xgb_params = params.get('best_xgb_params')
                             params_found_in_file = ebm_params and xgb_params
                         elif 'best_ebm_params' in best_params_all and 'best_xgb_params' in best_params_all:
                            # Check simpler structure, assuming it applies to the current phase
                            ebm_params = best_params_all['best_ebm_params']
                            xgb_params = best_params_all['best_xgb_params']
                            params_found_in_file = True

                         if not params_found_in_file:
                             print(f"Error: No compatible parameters found {source_info} for target {target}, index {target_index}, phase {args.phase}")
                             continue
                    else:
                        # Neither names nor file provided for --run-best
                         print("Error: --run-best requires either --best-ebm-param-name and --best-xgb-param-name OR --best-params-file.")
                         continue # Skip to next target/index

                    # If parameters were successfully loaded (either way)
                    if ebm_params and xgb_params:
                        print(f"Running with best parameters {source_info} for {target} index {target_index}, phase {args.phase}")
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
                            output_dir=args.output, # Will default if None
                            organized_dir=args.organized_dir,
                            target_index=target_index,
                            parallel=not args.no_parallel,
                            max_workers=args.max_workers,
                            sample=args.sample,
                            phase=args.phase # Pass phase
                        )
                    

                else:
                    # Run with default parameters (one set each)
                    print(f"Using default parameters for phase '{args.phase}' (no optimization or --run-best specified)")

                    # Select default parameter sets
                    ebm_params = ebm_parameter_sets['ebm_default']
                    xgb_params = xgb_residual_sets['xgb_residual_mse']

                    run_stacked_model_with_params(
                        target=target,
                        ebm_params=ebm_params,
                        xgb_params=xgb_params,
                        start_round=args.start,
                        end_round=args.end,
                        step=args.step,
                        output_dir=args.output, # Will default if None
                        organized_dir=args.organized_dir,
                        target_index=target_index,
                        parallel=not args.no_parallel,
                        max_workers=args.max_workers,
                        sample=args.sample,
                        phase=args.phase
                    )
        except Exception as e:
            print(f"Critical error processing target {target} for phase {args.phase}: {type(e).__name__} - {str(e)}")
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    main()