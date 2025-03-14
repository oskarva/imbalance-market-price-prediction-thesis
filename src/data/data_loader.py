"""
Data loading utilities for cross-validation with efficient memory management.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Generator


def get_cv_round_count() -> int:
    """
    Get the total number of cross-validation rounds available.
    
    Returns:
        int: The number of cross-validation rounds
    """
    # Look for X_test files to determine how many CV rounds we have
    data_dir = "./src/data/csv"
    files = [f for f in os.listdir(data_dir) if f.startswith('X_test_')]
    
    # Extract numbers from filenames
    round_numbers = [int(f.split('_')[-1].split('.')[0]) for f in files]
    
    if not round_numbers:
        raise FileNotFoundError("No cross-validation files found in data directory")
    
    return max(round_numbers) + 1  # +1 because we count from 0


def load_cv_round(cv_round: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a specific cross-validation round's data.
    
    Args:
        cv_round: The cross-validation round number to load
        
    Returns:
        Tuple containing (X_train, y_train, X_test, y_test) as pandas DataFrames
    """
    data_dir = "./src/data/csv"
    try:
        # Load X data with index_col=0 (timestamp as index)
        X_train = pd.read_csv(f"{data_dir}/X_train_{cv_round}.csv", index_col=0)
        X_test = pd.read_csv(f"{data_dir}/X_test_{cv_round}.csv", index_col=0)
        
        # For y data, we need to be careful - load without using index_col first to examine
        y_train_raw = pd.read_csv(f"{data_dir}/y_train_{cv_round}.csv")
        y_test_raw = pd.read_csv(f"{data_dir}/y_test_{cv_round}.csv")
        
        # Check column structure
        print(f"y_train_raw columns: {y_train_raw.columns.tolist()}")
        
        # Assuming first column is timestamp and second is the target
        if len(y_train_raw.columns) >= 2:
            # Standard case: first col is timestamp, others are targets
            timestamp_col = y_train_raw.columns[0]
            
            # Create proper y DataFrames with timestamp as index
            y_train = y_train_raw.set_index(timestamp_col)
            y_test = y_test_raw.set_index(timestamp_col)
        else:
            # If there's only one column, it might be that the target is saved in the index
            # Try loading again with index_col=0 but then extract index to a column
            y_train = pd.read_csv(f"{data_dir}/y_train_{cv_round}.csv", index_col=0)
            y_test = pd.read_csv(f"{data_dir}/y_test_{cv_round}.csv", index_col=0)
            
            # If y_train is empty, it means the target was indeed used as index
            if y_train.empty:
                print("Target appears to be in the index. Reconstructing DataFrame...")
                # Get the index name (or assign one if it's None)
                index_name = y_train.index.name or 'target'
                
                # Create a new DataFrame with the index as a column
                y_train = pd.DataFrame({index_name: y_train.index.values})
                y_test = pd.DataFrame({index_name: y_test.index.values})
        
        # Convert indices to datetime
        X_train.index = pd.to_datetime(X_train.index)
        X_test.index = pd.to_datetime(X_test.index)
        y_train.index = pd.to_datetime(y_train.index) if hasattr(y_train, 'index') else pd.to_datetime(y_train_raw[timestamp_col])
        y_test.index = pd.to_datetime(y_test.index) if hasattr(y_test, 'index') else pd.to_datetime(y_test_raw[timestamp_col])
        
        print(f"After processing - X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"y_train columns: {y_train.columns.tolist()}")
        
        return X_train, y_train, X_test, y_test
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not load cross-validation round {cv_round}: {str(e)}")
    

def cv_round_generator(start_round: int = 0, 
                      end_round: Optional[int] = None, 
                      step: int = 1) -> Generator:
    """
    Generator that yields cross-validation rounds one at a time to save memory.
    
    Args:
        start_round: The first round to yield (default: 0)
        end_round: The last round to yield (default: None, meaning all available rounds)
        step: Step size for iterating through rounds (default: 1)
        
    Yields:
        Tuple of (round_number, X_train, y_train, X_test, y_test)
    """
    if end_round is None:
        end_round = get_cv_round_count()
    
    for i in range(start_round, end_round, step):
        try:
            X_train, y_train, X_test, y_test = load_cv_round(i)
            yield i, X_train, y_train, X_test, y_test
        except FileNotFoundError as e:
            print(f"Warning: {str(e)}, skipping round {i}")
            continue


def preprocess_data(X_train: pd.DataFrame, 
                   y_train: pd.DataFrame,
                   X_test: pd.DataFrame, 
                   y_test: pd.DataFrame,
                   scale_features: bool = True,
                   remove_timezone: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data for model training.
    
    Args:
        X_train, y_train, X_test, y_test: Data from a cross-validation round
        scale_features: Whether to standardize features
        remove_timezone: Whether to remove timezone information from indices
        
    Returns:
        Preprocessed versions of X_train, y_train, X_test, y_test
    """
    # Make copies to avoid modifying the original data
    X_train_processed = X_train.copy()
    y_train_processed = y_train.copy()
    X_test_processed = X_test.copy()
    y_test_processed = y_test.copy()
    
    # Ensure indices are datetime type
    try:
        if not isinstance(X_train_processed.index, pd.DatetimeIndex):
            X_train_processed.index = pd.to_datetime(X_train_processed.index, utc=True)
        if not isinstance(y_train_processed.index, pd.DatetimeIndex):
            y_train_processed.index = pd.to_datetime(y_train_processed.index, utc=True)
        if not isinstance(X_test_processed.index, pd.DatetimeIndex):
            X_test_processed.index = pd.to_datetime(X_test_processed.index, utc=True)
        if not isinstance(y_test_processed.index, pd.DatetimeIndex):
            y_test_processed.index = pd.to_datetime(y_test_processed.index, utc=True)
    except Exception as e:
        print(f"Warning: Couldn't convert indices to datetime: {str(e)}")
        print("Continuing without timezone processing...")
        remove_timezone = False
    
    # Remove timezone info if requested and possible
    if remove_timezone:
        try:
            if hasattr(X_train_processed.index, 'tz') and X_train_processed.index.tz is not None:
                X_train_processed.index = X_train_processed.index.tz_localize(None)
            if hasattr(y_train_processed.index, 'tz') and y_train_processed.index.tz is not None:
                y_train_processed.index = y_train_processed.index.tz_localize(None)
            if hasattr(X_test_processed.index, 'tz') and X_test_processed.index.tz is not None:
                X_test_processed.index = X_test_processed.index.tz_localize(None)
            if hasattr(y_test_processed.index, 'tz') and y_test_processed.index.tz is not None:
                y_test_processed.index = y_test_processed.index.tz_localize(None)
        except Exception as e:
            print(f"Warning: Error removing timezone info: {str(e)}")
    
    # Scale features if requested
    if scale_features:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        # Fit on training data only
        X_train_values = X_train_processed.values
        scaler.fit(X_train_values)
        
        # Transform both training and test data
        X_train_processed = pd.DataFrame(
            scaler.transform(X_train_values),
            index=X_train_processed.index,
            columns=X_train_processed.columns
        )
        
        X_test_processed = pd.DataFrame(
            scaler.transform(X_test_processed.values),
            index=X_test_processed.index,
            columns=X_test_processed.columns
        )
    
    return X_train_processed, y_train_processed, X_test_processed, y_test_processed

def batch_generator(X: pd.DataFrame, 
                   y: pd.DataFrame, 
                   batch_size: int) -> Generator:
    """
    Generate batches for training to reduce memory usage during model training.
    
    Args:
        X: Feature DataFrame
        y: Target DataFrame
        batch_size: Number of samples per batch
        
    Yields:
        Tuple of (X_batch, y_batch) as numpy arrays
    """
    num_samples = len(X)
    indices = np.arange(num_samples)
    
    # Shuffle if this is training data
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Convert to numpy for efficiency with most ML models
        X_batch = X.iloc[batch_indices].values
        y_batch = y.iloc[batch_indices].values
        
        yield X_batch, y_batch

        
# Example usage
if __name__ == "__main__":
    # Show how many CV rounds we have
    num_rounds = get_cv_round_count()
    print(f"Found {num_rounds} cross-validation rounds")
    
    # Load and preprocess a specific round
    round_to_load = 0
    X_train, y_train, X_test, y_test = load_cv_round(round_to_load)
    print(f"Loaded round {round_to_load}:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Apply preprocessing
    X_train_proc, y_train_proc, X_test_proc, y_test_proc = preprocess_data(
        X_train, y_train, X_test, y_test, 
        scale_features=True, 
        remove_timezone=True
    )
    
    print("\nAfter preprocessing:")
    print(f"X_train timezone info: {X_train_proc.index.tz}")
    
    # Show first few rows of preprocessed data
    print("\nPreprocessed X_train sample:")
    print(X_train_proc.head())
