"""
Script to add time-based features to X files.
Creates new versions in a separate directory without modifying originals.
"""
import os
import pandas as pd
import numpy as np
import argparse
import re
from pathlib import Path

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
        
    # Time of day features (cyclic encoding to handle midnight/noon transitions)
    hours_in_day = 24
    df_with_features['sin_hour'] = np.sin(2 * np.pi * df_with_features.index.hour / hours_in_day)
    df_with_features['cos_hour'] = np.cos(2 * np.pi * df_with_features.index.hour / hours_in_day)

    # Month features (cyclic encoding to handle year transitions)
    months_in_year = 12
    df_with_features['sin_month'] = np.sin(2 * np.pi * df_with_features.index.month / months_in_year)
    df_with_features['cos_month'] = np.cos(2 * np.pi * df_with_features.index.month / months_in_year)
    
    return df_with_features

def process_x_files(source_dir, target_dir):
    """
    Process all X files in the source directory, add time features, and save to target directory.
    
    Args:
        source_dir: Directory containing original X files
        target_dir: Directory to save files with time features
    
    Returns:
        Tuple of (number of files processed, number of errors)
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all X files in the source directory
    x_files = [f for f in os.listdir(source_dir) if f.startswith('X_')]
    
    # Sort files by round number
    def extract_round(filename):
        match = re.search(r'_(\d+)\.csv$', filename)
        return int(match.group(1)) if match else float('inf')
    
    x_files.sort(key=extract_round)
    
    print(f"Found {len(x_files)} X files to process")
    
    # Process each file
    processed_count = 0
    error_count = 0
    
    for i, filename in enumerate(x_files):
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        try:
            # Read X file
            df = pd.read_csv(source_path, index_col=0)
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index, utc=True)
            
            # Add time features
            df_with_features = add_time_features(df)
            
            # Save to target directory
            df_with_features.to_csv(target_path)
            
            processed_count += 1
            
            # Print progress periodically
            if (i + 1) % 50 == 0 or i == len(x_files) - 1:
                print(f"Processed {i + 1}/{len(x_files)} files")
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            error_count += 1
    
    return processed_count, error_count

def main():
    parser = argparse.ArgumentParser(description='Add time features to X files')
    
    parser.add_argument('--source-dir', type=str, default='./src/data/organized/X_files',
                       help='Directory containing original X files')
    
    parser.add_argument('--target-dir', type=str, default='./src/data/organized/X_files_with_time_features',
                       help='Directory to save files with time features')
    
    args = parser.parse_args()
    
    print(f"Adding time features to X files from {args.source_dir}")
    print(f"Saving results to {args.target_dir}")
    
    processed_count, error_count = process_x_files(args.source_dir, args.target_dir)
    
    print(f"\nDone! Processed {processed_count} files with {error_count} errors")
    
    if processed_count > 0:
        # Get list of features in the processed files
        sample_file = os.listdir(args.target_dir)[0]
        sample_path = os.path.join(args.target_dir, sample_file)
        sample_df = pd.read_csv(sample_path)
        
        print("\nAdded the following time features:")
        # Get only the time features (the new columns)
        original_file = os.path.join(args.source_dir, sample_file)
        original_df = pd.read_csv(original_file)
        
        new_features = [col for col in sample_df.columns if col not in original_df.columns]
        
        for feature in new_features:
            print(f"  - {feature}")

if __name__ == "__main__":
    main()