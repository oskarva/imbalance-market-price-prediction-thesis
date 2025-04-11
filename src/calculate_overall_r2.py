import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate overall R² and other metrics across all CV predictions'
    )
    parser.add_argument(
        '--input-dir', '-i', type=str, required=True,
        help='Directory containing prediction CSV files'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default='./results',
        help='Directory to save results and visualizations (default: ./results)'
    )
    parser.add_argument(
        '--pattern', '-p', type=str, default='*predictions*.csv',
        help='Glob pattern to match prediction files (default: *predictions*.csv)'
    )
    parser.add_argument(
        '--round-regex', '-r', type=str, default=r'round(\d+)',
        help='Regex pattern to extract round number from filename (default: round(\\d+))'
    )
    parser.add_argument(
        '--actual-col', '-a', type=str, default='actual',
        help='Column name for actual values (default: actual)'
    )
    parser.add_argument(
        '--predicted-col', '-pr', type=str, default='predicted',
        help='Column name for predicted values (default: predicted)'
    )
    parser.add_argument(
        '--timestamp-col', '-t', type=str, default=None,
        help='Column name for timestamp values (optional)'
    )
    
    return parser.parse_args()

def load_prediction_files(
    input_dir: str, 
    pattern: str, 
    round_regex: str,
    actual_col: str,
    predicted_col: str
) -> Dict[int, pd.DataFrame]:
    """
    Load all prediction files from the input directory.
    
    Args:
        input_dir: Directory containing prediction files
        pattern: Glob pattern to match prediction files
        round_regex: Regex pattern to extract round number from filename
        actual_col: Column name for actual values
        predicted_col: Column name for predicted values
        
    Returns:
        Dictionary mapping round numbers to DataFrames
    """
    predictions = {}
    file_pattern = os.path.join(input_dir, pattern)
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"Error: No files found matching pattern '{file_pattern}'")
        sys.exit(1)
    
    print(f"Found {len(files)} prediction files")
    
    regex = re.compile(round_regex)
    
    for file in files:
        filename = os.path.basename(file)
        match = regex.search(filename)
        
        if match:
            round_num = int(match.group(1))
        else:
            # If round number can't be extracted, use file index
            round_num = len(predictions)
            
        try:
            df = pd.read_csv(file)
            
            # Verify required columns exist
            if actual_col not in df.columns:
                print(f"Warning: '{actual_col}' column not found in {filename}")
                continue
                
            if predicted_col not in df.columns:
                print(f"Warning: '{predicted_col}' column not found in {filename}")
                continue
            
            # Check for NaN values
            if df[actual_col].isna().any() or df[predicted_col].isna().any():
                nan_count = max(df[actual_col].isna().sum(), df[predicted_col].isna().sum())
                print(f"Warning: {nan_count} NaN values found in {filename}")
                
                # Drop NaN values
                df = df.dropna(subset=[actual_col, predicted_col])
                
            # Store DataFrame
            predictions[round_num] = df
            print(f"Loaded round {round_num} from {filename} with {len(df)} rows")
            
        except Exception as e:
            print(f"Error loading file {filename}: {str(e)}")
    
    if not predictions:
        print("Error: No valid prediction files loaded")
        sys.exit(1)
        
    return predictions

def calculate_overall_metrics(
    predictions: Dict[int, pd.DataFrame],
    actual_col: str,
    predicted_col: str
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate overall metrics across all predictions.
    
    Args:
        predictions: Dictionary mapping round numbers to DataFrames
        actual_col: Column name for actual values
        predicted_col: Column name for predicted values
        
    Returns:
        Tuple containing metrics dictionary and arrays of actual values, 
        predicted values, and errors
    """
    # Combine all predictions
    all_actual = []
    all_predicted = []
    
    for round_num, df in predictions.items():
        all_actual.extend(df[actual_col].values)
        all_predicted.extend(df[predicted_col].values)
    
    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    
    # Calculate errors
    errors = all_predicted - all_actual
    abs_errors = np.abs(errors)
    
    # Calculate metrics
    r2 = r2_score(all_actual, all_predicted)
    mae = mean_absolute_error(all_actual, all_predicted)
    rmse = np.sqrt(mean_squared_error(all_actual, all_predicted))
    correlation = np.corrcoef(all_actual, all_predicted)[0, 1]
    
    # Additional statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    error_std = np.std(errors)
    error_variance = np.var(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)
    
    # Percentiles
    p25_error = np.percentile(abs_errors, 25)
    p50_error = np.percentile(abs_errors, 50)
    p75_error = np.percentile(abs_errors, 75)
    p90_error = np.percentile(abs_errors, 90)
    p95_error = np.percentile(abs_errors, 95)
    p99_error = np.percentile(abs_errors, 99)
    
    # Check if min-max normalization would be helpful
    actual_range = np.max(all_actual) - np.min(all_actual)
    normalized_rmse = rmse / actual_range if actual_range != 0 else float('inf')
    
    # Create combined metrics dictionary
    metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'mean_error': mean_error,
        'median_error': median_error,
        'error_std': error_std,
        'error_variance': error_variance,
        'min_error': min_error,
        'max_error': max_error,
        'p25_error': p25_error,
        'p50_error': p50_error,
        'p75_error': p75_error,
        'p90_error': p90_error,
        'p95_error': p95_error,
        'p99_error': p99_error,
        'normalized_rmse': normalized_rmse,
        'sample_count': len(all_actual)
    }
    
    return metrics, all_actual, all_predicted, errors

def create_visualizations(
    all_actual: np.ndarray,
    all_predicted: np.ndarray,
    errors: np.ndarray,
    metrics: Dict[str, float],
    output_dir: str,
    timestamp_col: str = None,
    timestamps: np.ndarray = None
):
    """
    Create visualizations of the results.
    
    Args:
        all_actual: Array of actual values
        all_predicted: Array of predicted values
        errors: Array of prediction errors
        metrics: Dictionary of calculated metrics
        output_dir: Directory to save visualizations
        timestamp_col: Column name for timestamp values (optional)
        timestamps: Array of timestamp values (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set common style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 8))
    plt.scatter(all_actual, all_predicted, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(all_actual), np.min(all_predicted))
    max_val = max(np.max(all_actual), np.max(all_predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values (R² = {metrics["r2"]:.4f})')
    
    # Add text with key metrics
    text = f"R² = {metrics['r2']:.4f}\n"
    text += f"MAE = {metrics['mae']:.4f}\n"
    text += f"RMSE = {metrics['rmse']:.4f}\n"
    text += f"Correlation = {metrics['correlation']:.4f}\n"
    text += f"Sample Count = {metrics['sample_count']}"
    
    plt.figtext(0.15, 0.15, text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()
    
    # 2. Histogram of errors
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75, color='blue')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    
    # Add text with error statistics
    text = f"Mean Error = {metrics['mean_error']:.4f}\n"
    text += f"Median Error = {metrics['median_error']:.4f}\n"
    text += f"Error Std = {metrics['error_std']:.4f}\n"
    text += f"Min Error = {metrics['min_error']:.4f}\n"
    text += f"Max Error = {metrics['max_error']:.4f}"
    
    plt.figtext(0.15, 0.75, text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_histogram.png'), dpi=300)
    plt.close()
    
    # 3. Time series plot (if timestamps are available)
    if timestamps is not None and len(timestamps) == len(all_actual):
        plt.figure(figsize=(12, 8))
        
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        sorted_timestamps = timestamps[sorted_indices]
        sorted_actual = all_actual[sorted_indices]
        sorted_predicted = all_predicted[sorted_indices]
        
        plt.plot(sorted_timestamps, sorted_actual, 'b-', label='Actual')
        plt.plot(sorted_timestamps, sorted_predicted, 'r-', label='Predicted')
        
        plt.xlabel(f'Time ({timestamp_col})')
        plt.ylabel('Values')
        plt.title('Actual vs Predicted Values Over Time')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_series.png'), dpi=300)
        plt.close()
    
    # 4. Residual plot (errors vs predicted)
    plt.figure(figsize=(10, 6))
    plt.scatter(all_predicted, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Errors)')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'), dpi=300)
    plt.close()
    
    # 5. Q-Q plot to check error normality
    plt.figure(figsize=(8, 8))
    
    # Sort errors
    sorted_errors = np.sort(errors)
    
    # Generate theoretical quantiles
    n = len(sorted_errors)
    quantiles = np.arange(1, n + 1) / (n + 1)
    theoretical_quantiles = np.quantile(np.random.normal(0, 1, 10000), quantiles)
    
    # Standardize errors
    standardized_errors = (sorted_errors - np.mean(sorted_errors)) / np.std(sorted_errors)
    
    plt.scatter(theoretical_quantiles, standardized_errors, alpha=0.5)
    plt.plot([-3, 3], [-3, 3], 'r--')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title('Q-Q Plot of Errors')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qq_plot.png'), dpi=300)
    plt.close()

def save_results(metrics: Dict[str, float], output_dir: str):
    """
    Save results to CSV and text files.
    
    Args:
        metrics: Dictionary of calculated metrics
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Save as text file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("Overall Prediction Metrics\n")
        f.write("=========================\n\n")
        
        # Format metrics in categories
        f.write("Main Metrics:\n")
        f.write(f"R² Score: {metrics['r2']:.6f}\n")
        f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}\n")
        f.write(f"Correlation: {metrics['correlation']:.6f}\n")
        f.write(f"Normalized RMSE: {metrics['normalized_rmse']:.6f}\n\n")
        
        f.write("Error Statistics:\n")
        f.write(f"Mean Error: {metrics['mean_error']:.6f}\n")
        f.write(f"Median Error: {metrics['median_error']:.6f}\n")
        f.write(f"Error Std Deviation: {metrics['error_std']:.6f}\n")
        f.write(f"Error Variance: {metrics['error_variance']:.6f}\n")
        f.write(f"Minimum Error: {metrics['min_error']:.6f}\n")
        f.write(f"Maximum Error: {metrics['max_error']:.6f}\n\n")
        
        f.write("Error Percentiles (Absolute Error):\n")
        f.write(f"25th Percentile: {metrics['p25_error']:.6f}\n")
        f.write(f"50th Percentile (Median): {metrics['p50_error']:.6f}\n")
        f.write(f"75th Percentile: {metrics['p75_error']:.6f}\n")
        f.write(f"90th Percentile: {metrics['p90_error']:.6f}\n")
        f.write(f"95th Percentile: {metrics['p95_error']:.6f}\n")
        f.write(f"99th Percentile: {metrics['p99_error']:.6f}\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"Sample Count: {metrics['sample_count']}\n")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prediction files
    predictions = load_prediction_files(
        args.input_dir,
        args.pattern,
        args.round_regex,
        args.actual_col,
        args.predicted_col
    )
    
    # Calculate overall metrics
    metrics, all_actual, all_predicted, errors = calculate_overall_metrics(
        predictions,
        args.actual_col,
        args.predicted_col
    )
    
    # Extract timestamps if provided
    timestamps = None
    if args.timestamp_col:
        try:
            timestamps_list = []
            for round_num, df in predictions.items():
                if args.timestamp_col in df.columns:
                    timestamps_list.extend(df[args.timestamp_col].values)
            
            if timestamps_list:
                timestamps = np.array(timestamps_list)
        except Exception as e:
            print(f"Warning: Error processing timestamps: {str(e)}")
    
    # Create visualizations
    create_visualizations(
        all_actual,
        all_predicted,
        errors,
        metrics,
        args.output_dir,
        args.timestamp_col,
        timestamps
    )
    
    # Save results
    save_results(metrics, args.output_dir)
    
    # Print summary
    print("\nOverall Metrics:")
    print(f"R² Score: {metrics['r2']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Correlation: {metrics['correlation']:.6f}")
    print(f"Sample Count: {metrics['sample_count']}")
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
