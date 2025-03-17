#!/usr/bin/env python3
"""
naive_model.py - Apply a naive forecasting model to multiple datasets

This script applies a naive model to datasets in specified folders. The model
uses the actual value at time t as the prediction for times t+1 to t+32, then
uses the value at t+32 for predictions from t+33 to t+64, and so on.
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import argparse
import matplotlib.pyplot as plt


def find_train_files(base_dir, filename_pattern="y_train_572.csv"):
    """Find all training files in the specified directories."""
    # Recursively find all files matching the pattern
    search_path = os.path.join(base_dir, "**", filename_pattern)
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        raise ValueError(f"No files found in {base_dir} matching pattern {filename_pattern}")
    
    print(f"Found {len(files)} files:")
    for file in files:
        print(f"  - {file}")
    
    return files


def apply_naive_model(file_path, step_size=32, test_size=18012):
    """
    Apply the naive forecasting model to the specified file.
    
    Args:
        file_path: Path to the CSV file with target values
        step_size: Number of steps to keep the same prediction (default: 32)
        test_size: Number of timesteps to use from the end of the dataset (default: 18012)
        
    Returns:
        DataFrame with actual and predicted values
    """
    # Load the data
    try:
        df = pd.read_csv(file_path)
        
        # Set the datetime column to be the first column
        datetime_col = df.columns[0]
        
        # Ensure datetime column is properly parsed
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        except:
            print(f"Warning: Could not parse '{datetime_col}' as datetime in {file_path}")
        
        # Check if the data has a target column (second column since first is datetime)
        if len(df.columns) < 2:
            raise ValueError(f"File {file_path} does not have enough columns. Expected datetime and target.")
        
        if 'target' in df.columns:
            target_col = 'target'
        elif 'price' in df.columns:
            target_col = 'price'
        elif 'value' in df.columns:
            target_col = 'value'
        else:
            # Default to the second column
            target_col = df.columns[1]
            print(f"Warning: Using second column '{target_col}' as target in {file_path}")
        
        # Clean the target column - check for and handle NaN values
        if df[target_col].isna().any():
            print(f"Warning: Found {df[target_col].isna().sum()} NaN values in {target_col} column in {file_path}")
            
            # Fill NaN values using forward fill then backward fill
            df[target_col] = df[target_col].fillna(method='ffill').fillna(method='bfill')
            
            if df[target_col].isna().any():
                print(f"Error: Still have {df[target_col].isna().sum()} NaN values after filling")
                # If we still have NaNs, set them to 0
                df[target_col] = df[target_col].fillna(0)
            else:
                print(f"Successfully filled NaN values in {target_col}")
        
        # Get the length of the dataset
        data_length = len(df)
        
        # Calculate the starting index
        start_index = max(0, data_length - test_size)
        
        # Extract the test portion
        test_df = df.iloc[start_index:].copy().reset_index(drop=True)
        test_length = len(test_df)
        
        # Initialize predictions list
        predictions = []
        
        # Apply the naive model
        for i in range(0, test_length):
            # Determine which previous value to use
            # Find the most recent multiple of step_size before the current index
            if i == 0:
                # For the first prediction, use the last value from the training set
                if start_index > 0:
                    pred_value = df.iloc[start_index - 1][target_col]
                else:
                    # If there's no training data, use the first value of the test set
                    pred_value = test_df.iloc[0][target_col]
            else:
                # For subsequent predictions, use the value at the last step boundary
                last_step_idx = ((i - 1) // step_size) * step_size
                pred_value = test_df.iloc[last_step_idx][target_col]
                
            predictions.append(pred_value)
        
        # Add predictions to the test dataframe
        test_df['predicted'] = predictions
        test_df['actual'] = test_df[target_col]
        
        # Make sure the timestamp column is named 'timestamp' for compatibility with plotting script
        if datetime_col != 'timestamp':
            test_df['timestamp'] = test_df[datetime_col]
        
        return test_df, target_col, datetime_col
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None


def calculate_metrics(df, actual_col='actual', predicted_col='predicted'):
    """Calculate performance metrics for the predictions."""
    try:
        # Filter out rows where either actual or predicted has NaN
        valid_df = df.dropna(subset=[actual_col, predicted_col])
        
        if len(valid_df) < len(df):
            print(f"Warning: Dropped {len(df) - len(valid_df)} rows with NaN values when calculating metrics")
        
        if len(valid_df) == 0:
            print("Error: No valid rows left after dropping NaN values")
            return {
                'r2': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'records_count': len(df),
                'valid_records': 0
            }
        
        # Calculate R-squared
        r2 = r2_score(valid_df[actual_col], valid_df[predicted_col])
        
        # Calculate MAE and RMSE
        mae = np.mean(np.abs(valid_df[predicted_col] - valid_df[actual_col]))
        rmse = np.sqrt(np.mean((valid_df[predicted_col] - valid_df[actual_col]) ** 2))
        
        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'records_count': len(df),
            'valid_records': len(valid_df)
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'r2': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'records_count': len(df) if isinstance(df, pd.DataFrame) else 0,
            'valid_records': 0
        }


def main():
    """Main function to apply the naive model to all files."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply naive forecasting model to datasets")
    parser.add_argument("--base_dir", default="src/data/organized/", 
                        help="Base directory containing the dataset folders (default: src/data/organized/)")
    parser.add_argument("--step_size", type=int, default=32, 
                        help="Number of steps to keep the same prediction (default: 32)")
    parser.add_argument("--test_size", type=int, default=18012, 
                        help="Number of timesteps to use from the end of the dataset (default: 18012)")
    parser.add_argument("--output_dir", default="results/naive_model", 
                        help="Directory to save prediction results (default: results/naive_model)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all training files
    try:
        files = find_train_files(args.base_dir)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    
    # Dictionary to store results from all datasets
    all_results = {}
    
    # Process each file
    for file_path in files:
        # Extract folder name for reporting
        folder_name = os.path.basename(os.path.dirname(file_path))
        print(f"\nProcessing {folder_name} from {file_path}")
        
        # Apply the naive model
        result_df, target_col, datetime_col = apply_naive_model(
            file_path, 
            step_size=args.step_size,
            test_size=args.test_size
        )
        
        if result_df is not None:
            # Calculate metrics
            metrics = calculate_metrics(result_df)
            all_results[folder_name] = metrics
            
            # Print metrics
            print(f"Results for {folder_name}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
            
            # Save predictions to file
            output_file = os.path.join(args.output_dir, f"naive_predictions_{folder_name}.csv")
            result_df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
            
            # Create a simple plot to visualize the results
            # Skip if we're dealing with NaN values
            if np.isnan(result_df['actual']).all() or np.isnan(result_df['predicted']).all():
                print(f"Skipping plot for {folder_name} due to all NaN values")
                continue
                
            # Filter out NaN values for plotting
            plot_df = result_df.dropna(subset=['actual', 'predicted'])
            
            if len(plot_df) == 0:
                print(f"Skipping plot for {folder_name} due to no valid data after filtering NaNs")
                continue
                
            plt.figure(figsize=(15, 6))
            plt.plot(plot_df['timestamp'], plot_df['actual'], 'b-', label='Actual')
            plt.plot(plot_df['timestamp'], plot_df['predicted'], 'r--', label='Predicted')
            plt.legend()
            plt.title(f"Naive Model Predictions - {folder_name}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            # Format x-axis to show dates properly
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            
            # Save the figure
            plt_output_file = os.path.join(args.output_dir, f"naive_plot_{folder_name}.png")
            plt.savefig(plt_output_file)
            plt.close()
            print(f"Plot saved to {plt_output_file}")
    
    # Print a summary of all results
    print("\n===== SUMMARY OF RESULTS =====")
    
    # Create a summary dataframe for easy comparison
    summary_data = {
        'Dataset': [],
        'R²': [],
        'MAE': [],
        'RMSE': [],
        'Records': [],
        'Valid Records': []
    }
    
    for dataset, metrics in all_results.items():
        summary_data['Dataset'].append(dataset)
        summary_data['R²'].append(metrics['r2'])
        summary_data['MAE'].append(metrics['mae'])
        summary_data['RMSE'].append(metrics['rmse'])
        summary_data['Records'].append(metrics['records_count'])
        summary_data['Valid Records'].append(metrics.get('valid_records', metrics['records_count']))
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save the summary to a CSV file
    summary_file = os.path.join(args.output_dir, "naive_model_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()