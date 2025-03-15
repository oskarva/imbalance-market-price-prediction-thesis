#!/usr/bin/env python3
"""
plot_all_predictions.py - Visualize all actual vs predicted values across all cross-validation rounds

This script processes all prediction files in the specified directory, extracts actual and
predicted values, and creates a combined time series plot of all values in chronological order.
"""

import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def find_prediction_files(directory, pattern="*predictions*.csv"):
    """Find all prediction files in the specified directory."""
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    
    if not files:
        raise ValueError(f"No prediction files found in {directory} matching pattern {pattern}")
    
    # Extract round number from filenames
    def extract_round(filename):
        match = re.search(r'round(\d+)', os.path.basename(filename))
        if match:
            return int(match.group(1))
        return float('inf')  # Files without round number will be processed last
    
    # Sort files by round number
    files.sort(key=extract_round)
    
    print(f"Found {len(files)} prediction files")
    return files

def load_predictions(files, actual_col="actual", predicted_col="predicted", timestamp_col="timestamp"):
    """Load predictions from all files and combine them in order."""
    all_data = []
    
    for i, file in enumerate(files):
        try:
            df = pd.read_csv(file)
            
            # Verify required columns exist
            if actual_col not in df.columns:
                print(f"Warning: '{actual_col}' column not found in {file}, skipping")
                continue
                
            if predicted_col not in df.columns:
                print(f"Warning: '{predicted_col}' column not found in {file}, skipping")
                continue
            
            # Add timestamp if available
            if timestamp_col in df.columns:
                # Parse timestamps
                df['datetime'] = pd.to_datetime(df[timestamp_col])
                df = df.sort_values('datetime')
            else:
                # Create artificial timestamps if not available
                print(f"Warning: No timestamp column in {file}, using sequential indices")
                df['datetime'] = pd.date_range(start='2000-01-01', periods=len(df), freq='H')
            
            # Add round number
            df['round'] = i
            df['file'] = os.path.basename(file)
            
            all_data.append(df)
            print(f"Loaded {len(df)} rows from {os.path.basename(file)}")
            
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    if not all_data:
        raise ValueError("Failed to load any valid prediction data")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by datetime if available
    combined_df = combined_df.sort_values('datetime')
    
    return combined_df

def plot_all_predictions(df, actual_col="actual", predicted_col="predicted", output_file="all_predictions.png"):
    """Create a time series plot of all actual vs predicted values."""
    plt.figure(figsize=(15, 8))
    
    # Plot actual and predicted values
    plt.plot(df.index, df[actual_col], 'b-', label='Actual', linewidth=1.5)
    plt.plot(df.index, df[predicted_col], 'r--', label='Predicted', linewidth=1)
    
    # Add vertical lines between rounds
    for round_end in df[df['round'].diff() == 1].index:
        plt.axvline(x=round_end, color='gray', linestyle='-', alpha=0.3)
    
    # Customize plot
    plt.xlabel('Time Sequence')
    plt.ylabel('Price')
    plt.title('All Rounds - Actual vs Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    # Create a second plot for correlation
    plt.figure(figsize=(10, 10))
    plt.scatter(df[actual_col], df[predicted_col], alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(df[actual_col].min(), df[predicted_col].min())
    max_val = max(df[actual_col].max(), df[predicted_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'g--')
    
    # Calculate overall metrics
    r2 = np.corrcoef(df[actual_col], df[predicted_col])[0, 1] ** 2
    mae = np.mean(np.abs(df[predicted_col] - df[actual_col]))
    rmse = np.sqrt(np.mean((df[predicted_col] - df[actual_col]) ** 2))
    
    # Add metrics to plot
    plt.annotate(f"R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}",
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Overall Correlation Plot - Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Save the correlation plot
    corr_output_file = "overall_correlation.png"
    plt.tight_layout()
    plt.savefig(corr_output_file, dpi=300)
    print(f"Correlation plot saved to {corr_output_file}")

    # Create additional plots for different views of the data
    # 1. Error distribution histogram
    plt.figure(figsize=(12, 6))
    errors = df[predicted_col] - df[actual_col]
    plt.hist(errors, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("error_distribution.png", dpi=300)
    
    # 2. Error over time
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, errors, 'g-', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time Sequence')
    plt.ylabel('Error (Predicted - Actual)')
    plt.title('Prediction Error Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("error_over_time.png", dpi=300)
    
    # Return statistics
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'records_count': len(df)
    }

def main():
    """Main function to load and plot all predictions."""
    # Set the directory containing prediction files
    data_dir = "results/cv_run_20250314-132254"
    
    # Find all prediction files
    try:
        files = find_prediction_files(data_dir)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    
    # Load predictions
    try:
        df = load_predictions(files)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    
    # Plot predictions
    stats = plot_all_predictions(df)
    
    # Print statistics
    print("\nOverall Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

if __name__ == "__main__":
    main()