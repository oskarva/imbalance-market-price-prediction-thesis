#!/usr/bin/env python3
"""
plot_predictions.py - Visualize actual vs predicted values from a single CSV file

This script processes a single predictions CSV file, extracts actual and predicted values,
and creates visualizations to analyze model performance.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_predictions(file_path, actual_col="actual", predicted_col="predicted", timestamp_col="timestamp"):
    """Load predictions from a single CSV file."""
    try:
        df = pd.read_csv(file_path)
        
        # Verify required columns exist
        if actual_col not in df.columns:
            raise ValueError(f"'{actual_col}' column not found in {file_path}")
                
        if predicted_col not in df.columns:
            raise ValueError(f"'{predicted_col}' column not found in {file_path}")
        
        # Process timestamps if available
        if timestamp_col in df.columns:
            # Parse timestamps
            df['datetime'] = pd.to_datetime(df[timestamp_col])
            # Sort data by timestamps
            df = df.sort_values('datetime')
        else:
            # Create artificial timestamps if not available
            print(f"Warning: No timestamp column in {file_path}, using sequential indices")
            df['datetime'] = pd.date_range(start='2000-01-01', periods=len(df), freq='H')
        
        # Add a sequential index for plotting
        df['seq_index'] = range(len(df))
        
        print(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")
        return df
            
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {str(e)}")

def plot_predictions(df, actual_col="actual", predicted_col="predicted", output_prefix="prediction"):
    """Create visualizations for model prediction analysis."""
    # 1. Time series plot
    plt.figure(figsize=(15, 8))
    
    # Plot actual and predicted values
    plt.plot(df['seq_index'], df[actual_col], 'b-', linewidth=1.5, label='Actual')
    plt.plot(df['seq_index'], df[predicted_col], 'r--', linewidth=1, label='Predicted')
    
    # Customize plot
    plt.xlabel('Time Sequence')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a tight layout
    plt.tight_layout()
    
    # Save the figure
    timeseries_file = f"{output_prefix}_timeseries.png"
    plt.savefig(timeseries_file, dpi=300)
    print(f"Time series plot saved to {timeseries_file}")
    
    # 2. Correlation plot
    plt.figure(figsize=(10, 10))
    plt.scatter(df[actual_col], df[predicted_col], alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(df[actual_col].min(), df[predicted_col].min())
    max_val = max(df[actual_col].max(), df[predicted_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'g--')
    
    # Calculate metrics
    try:
        r2 = np.corrcoef(df[actual_col], df[predicted_col])[0, 1] ** 2
        # Check if r2 is NaN (can happen with constant values)
        if np.isnan(r2):
            r2 = 0
    except:
        r2 = 0  # In case of error, default to 0
        
    mae = np.mean(np.abs(df[predicted_col] - df[actual_col]))
    rmse = np.sqrt(np.mean((df[predicted_col] - df[actual_col]) ** 2))
    
    # Add metrics to plot
    plt.annotate(f"R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}",
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Correlation Plot - Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Save the correlation plot
    corr_output_file = f"{output_prefix}_correlation.png"
    plt.tight_layout()
    plt.savefig(corr_output_file, dpi=300)
    print(f"Correlation plot saved to {corr_output_file}")

    # 3. Error distribution histogram
    plt.figure(figsize=(12, 6))
    errors = df[predicted_col] - df[actual_col]
    plt.hist(errors, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    error_dist_file = f"{output_prefix}_error_distribution.png"
    plt.savefig(error_dist_file, dpi=300)
    print(f"Error distribution plot saved to {error_dist_file}")
    
    # 4. Error over time
    plt.figure(figsize=(15, 6))
    plt.plot(df['seq_index'], df[predicted_col] - df[actual_col], 'g-', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time Sequence')
    plt.ylabel('Error (Predicted - Actual)')
    plt.title('Prediction Error Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    error_time_file = f"{output_prefix}_error_over_time.png"
    plt.savefig(error_time_file, dpi=300)
    print(f"Error over time plot saved to {error_time_file}")
    
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
    """Main function to load and plot predictions."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualize actual vs predicted values from a single CSV file")
    parser.add_argument("file_path", help="Path to the predictions CSV file")
    parser.add_argument("--actual", default="actual", help="Column name for actual values (default: 'actual')")
    parser.add_argument("--predicted", default="predicted", help="Column name for predicted values (default: 'predicted')")
    parser.add_argument("--timestamp", default="timestamp", help="Column name for timestamps (default: 'timestamp')")
    parser.add_argument("--output", default="prediction", help="Prefix for output files (default: 'prediction')")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.file_path):
        print(f"Error: File '{args.file_path}' not found")
        return
    
    # Load predictions
    try:
        df = load_predictions(args.file_path, args.actual, args.predicted, args.timestamp)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    
    # Plot predictions
    stats = plot_predictions(df, args.actual, args.predicted, args.output)
    
    # Print statistics
    print("\nOverall Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

if __name__ == "__main__":
    main()