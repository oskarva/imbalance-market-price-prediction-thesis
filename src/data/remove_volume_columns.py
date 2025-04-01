import os
import glob
import pandas as pd
import csv
from datetime import datetime
import argparse

def process_csv_files(base_dir="src/data/csv", create_backup=True, dry_run=False, verbose=False):
    """
    Process all CSV files in the specified directory structure,
    removing the regulation up and down columns.
    
    Parameters:
    base_dir (str): The base directory containing the noX folders
    create_backup (bool): Whether to create backup files before modifying
    dry_run (bool): If True, only report actions but don't modify files
    verbose (bool): If True, print more detailed information
    """
    
    # Create a timestamp for backup folders
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(base_dir, f"backup_{timestamp}")
    
    if create_backup and not dry_run:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Created backup directory: {backup_dir}")
    
    # Get all directories matching the pattern 'noX'
    area_dirs = glob.glob(os.path.join(base_dir, "no*"))
    
    total_files = 0
    modified_files = 0
    
    for area_dir in area_dirs:
        # Extract the area code (e.g., 'no1', 'no2')
        area_code = os.path.basename(area_dir)
        print(f"Processing area: {area_code}")
        
        # Create area-specific backup directory
        area_backup_dir = os.path.join(backup_dir, area_code)
        if create_backup and not dry_run:
            os.makedirs(area_backup_dir, exist_ok=True)
        
        # Find all CSV files in this directory matching the pattern
        csv_patterns = [
            "X_train_*.csv",
            "y_train_*.csv",
            "X_test_*.csv",
            "y_test_*.csv"
        ]
        
        csv_files = []
        for pattern in csv_patterns:
            csv_files.extend(glob.glob(os.path.join(area_dir, pattern)))
        
        for csv_file in csv_files:
            total_files += 1
            try:
                filename = os.path.basename(csv_file)
                print(f"  Processing file: {filename}")
                
                # First, read the header to identify columns
                with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                
                if verbose:
                    print(f"    Original header: {header}")
                
                # Define the columns to remove, using the area code
                patterns_to_remove = [
                    f"vol {area_code} regulation up mwh cet h a",
                    f"vol {area_code} regulation down mwh cet h a"
                ]
                
                # Find the indices of columns to remove
                indices_to_remove = []
                for i, col_name in enumerate(header):
                    if col_name in patterns_to_remove:
                        indices_to_remove.append(i)
                        print(f"    Found column to remove at index {i}: '{col_name}'")
                
                if indices_to_remove:
                    # Create backup if requested
                    if create_backup and not dry_run:
                        backup_file = os.path.join(area_backup_dir, filename)
                        if not dry_run:
                            # Simple file copy for backup
                            with open(csv_file, 'r', newline='', encoding='utf-8') as src, \
                                 open(backup_file, 'w', newline='', encoding='utf-8') as dst:
                                dst.write(src.read())
                            print(f"    Created backup at: {backup_file}")
                    
                    if not dry_run:
                        # Read all rows
                        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            all_rows = list(reader)
                        
                        # Create a new header and rows by removing the specified columns
                        new_header = [col for i, col in enumerate(header) if i not in indices_to_remove]
                        new_rows = []
                        for row in all_rows[1:]:  # Skip header
                            new_row = [val for i, val in enumerate(row) if i not in indices_to_remove]
                            new_rows.append(new_row)
                        
                        # Write the file back
                        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(new_header)
                            writer.writerows(new_rows)
                        
                        print(f"    Saved modified file with {len(new_header)} columns")
                        if verbose:
                            print(f"    New header: {new_header}")
                        
                        modified_files += 1
                    else:
                        print(f"    (Dry run - columns would be removed: {[header[i] for i in indices_to_remove]})")
                else:
                    print(f"    No matching columns found to remove.")
                    
            except Exception as e:
                print(f"    Error processing {csv_file}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print(f"Processing complete! Processed {total_files} files, modified {modified_files} files.")
    if dry_run:
        print("This was a dry run - no files were actually modified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove regulation columns from CSV files.')
    parser.add_argument('--base-dir', default="src/data/csv", help='Base directory containing the noX folders')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--verbose', action='store_true', help='Print more detailed information')
    
    args = parser.parse_args()
    
    process_csv_files(
        base_dir=args.base_dir,
        create_backup=not args.no_backup,
        dry_run=args.dry_run,
        verbose=args.verbose
    )