"""
Simple script to create y files from extracted target data with proper numeric sorting.
"""
import os
import pandas as pd
import argparse
import re
import glob

def create_y_files(extracted_dir, organized_dir, target_file=None):
    """
    Create y_train and y_test files from extracted target data.
    
    Args:
        extracted_dir: Directory containing extracted target data
        organized_dir: Base directory for organized files
        target_file: Specific target file to process (optional)
    """
    # Create X_files directory if it doesn't exist
    x_files_dir = os.path.join(organized_dir, "X_files")
    
    # Get the list of target files
    if target_file:
        # Process specific file
        target_files = [os.path.join(extracted_dir, target_file)]
    else:
        # Process all CSV files in the extracted directory
        target_files = glob.glob(os.path.join(extracted_dir, "*.csv"))
    
    print(f"Found {len(target_files)} target files")
    
    for target_path in target_files:
        target_file = os.path.basename(target_path)
        target_name = os.path.splitext(target_file)[0]
        
        print(f"\nProcessing target: {target_name}")
        
        # Create output directory
        output_dir = os.path.join(organized_dir, target_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load target data
        try:
            target_df = pd.read_csv(target_path, index_col=0)
            target_df.index = pd.to_datetime(target_df.index)
            print(f"Loaded {len(target_df)} rows of target data")
        except Exception as e:
            print(f"Error loading target file: {str(e)}")
            continue
        
        # Get all X_train and X_test files
        x_train_files = [f for f in os.listdir(x_files_dir) if f.startswith('X_train_')]
        x_test_files = [f for f in os.listdir(x_files_dir) if f.startswith('X_test_')]
        
        # Extract round numbers and convert to integers for proper sorting
        train_rounds = []
        for x_file in x_train_files:
            match = re.search(r'_(\d+)\.csv$', x_file)
            if match:
                round_num = int(match.group(1))
                train_rounds.append((round_num, x_file))
                
        test_rounds = []
        for x_file in x_test_files:
            match = re.search(r'_(\d+)\.csv$', x_file)
            if match:
                round_num = int(match.group(1))
                test_rounds.append((round_num, x_file))
        
        # Sort by round number (numeric sorting)
        train_rounds.sort()
        test_rounds.sort()
        
        train_count = 0
        test_count = 0
        
        # Process X_train files in correct numeric order
        for round_num, x_file in train_rounds:
            try:
                # Load X_train
                x_df = pd.read_csv(os.path.join(x_files_dir, x_file), index_col=0)
                x_df.index = pd.to_datetime(x_df.index)
                
                # Get matching target data
                common_indices = x_df.index.intersection(target_df.index)
                
                if len(common_indices) == 0:
                    print(f"Warning: No matching data for round {round_num}")
                    continue
                
                # Create aligned y_train
                y_train = target_df.loc[common_indices]
                
                # Save y_train
                y_train_path = os.path.join(output_dir, f"y_train_{round_num}.csv")
                y_train.to_csv(y_train_path)
                
                train_count += 1
                
                if train_count % 50 == 0:
                    print(f"Created {train_count} y_train files")
            except Exception as e:
                print(f"Error processing X_train_{round_num}: {str(e)}")
        
        # Process X_test files in correct numeric order
        for round_num, x_file in test_rounds:
            try:
                # Load X_test
                x_df = pd.read_csv(os.path.join(x_files_dir, x_file), index_col=0)
                x_df.index = pd.to_datetime(x_df.index)
                
                # Get matching target data
                common_indices = x_df.index.intersection(target_df.index)
                
                if len(common_indices) == 0:
                    print(f"Warning: No matching data for round {round_num}")
                    continue
                
                # Create aligned y_test
                y_test = target_df.loc[common_indices]
                
                # Save y_test
                y_test_path = os.path.join(output_dir, f"y_test_{round_num}.csv")
                y_test.to_csv(y_test_path)
                
                test_count += 1
                
                if test_count % 50 == 0:
                    print(f"Created {test_count} y_test files")
            except Exception as e:
                print(f"Error processing X_test_{round_num}: {str(e)}")
        
        print(f"Created {train_count} y_train files and {test_count} y_test files for {target_name}")

def main():
    parser = argparse.ArgumentParser(description='Create y files from extracted target data')
    
    parser.add_argument('--extracted-dir', type=str, default='./src/data/extracted',
                       help='Directory containing extracted target data')
    
    parser.add_argument('--organized-dir', type=str, default='./src/data/organized',
                       help='Base directory for organized files')
    
    parser.add_argument('--target', type=str, default=None,
                       help='Specific target file to process')
    
    args = parser.parse_args()
    
    create_y_files(
        extracted_dir=args.extracted_dir,
        organized_dir=args.organized_dir,
        target_file=args.target
    )

if __name__ == "__main__":
    main()