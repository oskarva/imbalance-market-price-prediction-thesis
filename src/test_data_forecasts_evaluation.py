import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import argparse
import re

def find_files(base_dir, train_pattern="X_train_572.csv", test_pattern="X_test_*.csv"):
    """Find train and test files in the specified directory."""
    # Find the train file
    train_path = os.path.join(base_dir, train_pattern)
    train_files = glob.glob(train_path)
    
    if not train_files:
        raise ValueError(f"Training file not found: {train_path}")
    
    train_file = train_files[0]
    print(f"Found training file: {train_file}")
    
    # Find all test files
    test_path = os.path.join(base_dir, test_pattern)
    test_files = glob.glob(test_path)
    
    if not test_files:
        raise ValueError(f"No test files found matching: {test_path}")
    
    # Extract round number from filenames
    def extract_round(filename):
        match = re.search(r'X_test_(\d+)\.csv', os.path.basename(filename))
        if match:
            return int(match.group(1))
        return float('inf')  # Files without valid round number will be processed last
    
    # Sort files by round number
    test_files.sort(key=extract_round)
    
    print(f"Found {len(test_files)} test files")
    return train_file, test_files

def load_and_compare(train_file, test_files, output_dir):
    """Load train and test files, compare data, and calculate metrics."""
    # Load the training data (actual values)
    try:
        train_df = pd.read_csv(train_file)
        
        # Make sure the first column is a datetime column
        datetime_col = train_df.columns[0]
        try:
            train_df[datetime_col] = pd.to_datetime(train_df[datetime_col])
        except:
            print(f"Warning: Could not parse '{datetime_col}' as datetime in {train_file}")
        
        print(f"Loaded training data with {len(train_df)} rows and columns: {train_df.columns.tolist()}")
    except Exception as e:
        raise ValueError(f"Error loading training file {train_file}: {str(e)}")
    
    # Dictionary to store all predictions by column
    all_predictions = {}
    
    # Initialize the dictionary with all columns (except datetime)
    for col in train_df.columns[1:]:
        all_predictions[col] = {
            'timestamp': [],
            'actual': [],
            'predicted': []
        }
    
    # Process each test file
    for test_file in test_files:
        try:
            # Extract the test number from the filename
            test_num = re.search(r'X_test_(\d+)\.csv', os.path.basename(test_file))
            if test_num:
                test_num = int(test_num.group(1))
            else:
                test_num = -1
            
            test_df = pd.read_csv(test_file)
            
            # Make sure the first column is a datetime column
            test_datetime_col = test_df.columns[0]
            try:
                test_df[test_datetime_col] = pd.to_datetime(test_df[test_datetime_col])
            except:
                print(f"Warning: Could not parse '{test_datetime_col}' as datetime in {test_file}")
            
            print(f"Processing test file {os.path.basename(test_file)} with {len(test_df)} rows")
            
            # For each row in the test file
            for idx, test_row in test_df.iterrows():
                test_timestamp = test_row[test_datetime_col]
                
                # Find the corresponding row in the train file
                train_row = train_df[train_df[datetime_col] == test_timestamp]
                
                if train_row.empty:
                    print(f"Warning: No matching timestamp {test_timestamp} in training data")
                    continue
                
                # Compare each column (except the first datetime column)
                for col in test_df.columns[1:]:
                    if col not in train_df.columns:
                        # Skip columns that don't exist in training data
                        continue
                    
                    # Add the values to our predictions dictionary
                    all_predictions[col]['timestamp'].append(test_timestamp)
                    all_predictions[col]['actual'].append(train_row[col].iloc[0])
                    all_predictions[col]['predicted'].append(test_row[col])
            
        except Exception as e:
            print(f"Error processing test file {test_file}: {str(e)}")
            continue
    
    # Calculate metrics and create plots for each column
    results = {}
    
    for col, data in all_predictions.items():
        # Skip columns with no predictions
        if not data['predicted']:
            print(f"No predictions found for column: {col}")
            continue
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame({
            'timestamp': data['timestamp'],
            'actual': data['actual'],
            'predicted': data['predicted']
        })
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        if len(df_clean) < len(df):
            print(f"Warning: Dropped {len(df) - len(df_clean)} rows with NaN values for column {col}")
        
        if len(df_clean) == 0:
            print(f"Error: All data for column {col} contains NaN values")
            results[col] = {
                'r2': np.nan,
                'records_count': len(df),
                'valid_records': 0
            }
            continue
        
        # Sort by timestamp for meaningful time series plots
        df_clean = df_clean.sort_values('timestamp')
        
        # Calculate R-squared
        r2 = r2_score(df_clean['actual'], df_clean['predicted'])
        
        results[col] = {
            'r2': r2,
            'records_count': len(df),
            'valid_records': len(df_clean)
        }
        
        # Create a time series plot
        plt.figure(figsize=(15, 6))
        plt.plot(df_clean['timestamp'], df_clean['actual'], 'b-', label='Actual')
        plt.plot(df_clean['timestamp'], df_clean['predicted'], 'r--', label='Predicted')
        plt.legend()
        plt.title(f"Forecast Evaluation - {col}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        
        # Add R² annotation to the plot
        plt.annotate(f"R² = {r2:.4f}",
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        # Save the figure
        safe_col_name = col.replace('/', '_').replace('\\', '_').replace(' ', '_').replace(':', '_')
        plt_output_file = os.path.join(output_dir, f"forecast_eval_{safe_col_name}.png")
        plt.savefig(plt_output_file)
        plt.close()
        print(f"Plot saved to {plt_output_file}")
        
        # Save the predictions to CSV
        csv_output_file = os.path.join(output_dir, f"forecast_data_{safe_col_name}.csv")
        df_clean.to_csv(csv_output_file, index=False)
        print(f"Data saved to {csv_output_file}")
    
    return results

def main():
    """Main function to compare forecasts to actual values."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compare forecasted values with actual values")
    parser.add_argument("--base_dir", default="src/data/organized/X_files/", 
                        help="Directory containing X_train and X_test files")
    parser.add_argument("--output_dir", default="results/forecast_evaluation", 
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find train and test files
    try:
        train_file, test_files = find_files(args.base_dir)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    
    # Load and compare data
    try:
        results = load_and_compare(train_file, test_files, args.output_dir)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    
    # Print a summary of all results
    print("\n===== SUMMARY OF RESULTS =====")
    
    # Create a summary dataframe for easy comparison
    summary_data = {
        'Column': [],
        'R²': [],
        'Records': [],
        'Valid Records': []
    }
    
    for column, metrics in results.items():
        summary_data['Column'].append(column)
        summary_data['R²'].append(metrics['r2'])
        summary_data['Records'].append(metrics['records_count'])
        summary_data['Valid Records'].append(metrics['valid_records'])
    
    # Convert to DataFrame and sort by R² (descending)
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('R²', ascending=False)
    print(summary_df.to_string(index=False))
    
    # Save the summary to a CSV file
    summary_file = os.path.join(args.output_dir, "forecast_evaluation_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()