import volue_insight_timeseries
from volue_insight_timeseries.curves import InstanceCurve
import pandas as pd
import numpy as np
from .feature_engineering import *
import os
import time
import gc # Import gc for memory management

def get_data(X_curve_names, y_curve_names, sub_area, session,
             start_date, val_start_date, test_start_date, end_date, # Date-based periods
             output_base_path,
             X_to_forecast, # No None values allowed
             crossval_horizon=32, batch_size=10,
             add_time=False, add_lag=False, add_rolling=False,
             include_y_in_X=False, lag_value=32):
    """
    Retrieves data, performs feature engineering, and creates two sets of
    rolling-origin cross-validation files: one for a validation period and
    one for a test period, defined by dates. Saves sets to phase-specific
    subfolders.

    Parameters:
    -----------
    X_curve_names, y_curve_names: Lists of curve names.
    sub_area: Sub-area identifier for subfolder name.
    session: API session.
    start_date: Overall start date for data fetching (Timestamp, initially potentially naive).
    val_start_date: Start timestamp defining the validation CV period (Timestamp, initially potentially naive).
    test_start_date: Start timestamp defining the test CV period (Timestamp, initially potentially naive).
    end_date: Overall end date for data fetching (defines end of test CV period) (Timestamp, initially potentially naive).
    output_base_path: Base directory for saving CSV folders.
    X_to_forecast: Dict mapping columns to external forecast curve names (strings).
    crossval_horizon: Number of steps to predict in each CV round (default: 32).
    batch_size: Number of rounds per memory cleanup batch.
    add_time, add_lag, add_rolling, include_y_in_X, lag_value: Feature engineering options.

    Returns:
    --------
    tuple: (X_cols, y_cols, val_rounds_info, test_rounds_info) - Column names and dicts
           containing calculated start/end rounds for validation and test phases.
           Returns (None, None, {}, {}) on critical error.
    """
    # Ensure dates are Timestamps (initial parsing happens in main script, passed here)
    # Basic type check
    if not all(isinstance(d, pd.Timestamp) for d in [start_date, val_start_date, test_start_date, end_date]):
        print("Error: Input dates must be pandas Timestamps.")
        return None, None, {}, {}

    # Initial validation of date order (will be re-checked after potential timezone changes)
    try:
        # Comparison might fail here if inputs have mixed timezones already, but main check is later
        if not start_date < val_start_date < test_start_date < end_date:
             print("Warning: Initial date order check (start < val < test < end) failed. Will re-evaluate after timezone sync.")
             # Don't raise error yet, let timezone sync happen first
    except TypeError:
         print("Warning: Initial date comparison failed due to mixed timezones. Will re-evaluate after sync.")
         # Don't raise error yet

    # --- Data Fetching and Initial Cleaning ---
    print(f"Fetching data from {start_date} to {end_date}...")
    combined_curves = list(set(X_curve_names + y_curve_names))
    try:
        # Assume _get_data returns a DataFrame with potentially tz-aware index
        cleaned_df = _get_data(combined_curves, y_curve_names, session, start_date, end_date)
        if cleaned_df.empty: print("Error: Fetched data empty after cleaning."); return None, None, {}, {}
        print(f"Initial data shape after NaN row removal: {cleaned_df.shape}")
        print(f"Initial data range: {cleaned_df.index.min()} to {cleaned_df.index.max()}")

        # ---> START FIX: Synchronize Timezone Awareness <---
        data_tz = cleaned_df.index.tz
        if data_tz:
            print(f"Data index is timezone-aware ({data_tz}). Applying timezone to input dates.")
            try:
                # Localize or convert the input timestamps to match the data's timezone
                if start_date.tzinfo is None: start_date = start_date.tz_localize(data_tz)
                elif start_date.tzinfo != data_tz: start_date = start_date.tz_convert(data_tz)
                # else: already correct tz

                if val_start_date.tzinfo is None: val_start_date = val_start_date.tz_localize(data_tz)
                elif val_start_date.tzinfo != data_tz: val_start_date = val_start_date.tz_convert(data_tz)

                if test_start_date.tzinfo is None: test_start_date = test_start_date.tz_localize(data_tz)
                elif test_start_date.tzinfo != data_tz: test_start_date = test_start_date.tz_convert(data_tz)

                if end_date.tzinfo is None: end_date = end_date.tz_localize(data_tz)
                elif end_date.tzinfo != data_tz: end_date = end_date.tz_convert(data_tz)

                print(f"  Corrected start_date: {start_date}")
                print(f"  Corrected val_start_date: {val_start_date}")
                print(f"  Corrected test_start_date: {test_start_date}")
                print(f"  Corrected end_date: {end_date}")

                # Re-validate date order after timezone synchronization
                if not start_date < val_start_date < test_start_date < end_date:
                    raise ValueError(f"Dates are not correctly ordered after timezone sync: "
                                     f"start={start_date}, val={val_start_date}, "
                                     f"test={test_start_date}, end={end_date}")

            except Exception as e:
                 print(f"Error applying timezone ({data_tz}) or validating date order post-sync: {e}")
                 return None, None, {}, {}

        elif any(d.tzinfo is not None for d in [start_date, val_start_date, test_start_date, end_date]):
            # Handle the case where data is naive but inputs somehow became aware
            print("Warning: Data is tz-naive, but input dates are tz-aware. Making inputs naive.")
            start_date = start_date.tz_localize(None)
            val_start_date = val_start_date.tz_localize(None)
            test_start_date = test_start_date.tz_localize(None)
            end_date = end_date.tz_localize(None)
            # Re-validate date order after making naive
            if not start_date < val_start_date < test_start_date < end_date:
                 raise ValueError("Dates are not correctly ordered after making them naive.")

        # Else: Both data index and input dates are naive - no action needed, initial order check stands if it passed.
        # Re-check initial order if it wasn't checked properly before due to TypeError
        elif not start_date < val_start_date < test_start_date < end_date:
             raise ValueError("Dates must be ordered: start < val_start < test_start < end")


        # ---> END FIX <---

    except Exception as e:
        print(f"Error during data fetching/cleaning or timezone sync: {e}")
        return None, None, {}, {}


    # --- Feature Engineering ---
    # df will inherit the timezone awareness from cleaned_df
    df = cleaned_df.copy(); print("Applying feature engineering...")
    if add_time: df = add_time_features(df); print("Added time features.")
    original_cols = df.columns.tolist()
    if add_lag:
        lag_cols_added = []
        for col in combined_curves:
            if col in df.columns: lag_col_name = f"{col}_lag_{lag_value}"; df[lag_col_name] = df[col].shift(lag_value); lag_cols_added.append(lag_col_name)
        print(f"Added lag features for {len(lag_cols_added)} columns.")
    if add_rolling:
        rolling_cols_added = []
        cols_to_roll = [col for col in df.columns if f"_lag_{lag_value}" in col]
        if not cols_to_roll: cols_to_roll = [col for col in combined_curves if col in df.columns]; print(f"Rolling original cols: {cols_to_roll}")
        else: print(f"Rolling lag cols: {cols_to_roll}")
        if cols_to_roll:
            rolling_df = create_rolling_features(df[cols_to_roll], columns=cols_to_roll)
            for col in rolling_df.columns:
                if col not in cols_to_roll: df[col] = rolling_df[col]; rolling_cols_added.append(col)
            print(f"Added {len(rolling_cols_added)} rolling features.")

    initial_len = len(df); df = df.dropna()
    print(f"Shape after feature engineering & dropna: {df.shape}")
    print(f"Dropped {initial_len - len(df)} rows from feature engineering NaNs.")
    if df.empty: print("Error: DataFrame empty after feature engineering."); return None, None, {}, {}

    # final_start_date and final_end_date will have same tz awareness as df.index
    final_start_date = df.index.min(); final_end_date = df.index.max()
    print(f"Data range after feature engineering: {final_start_date} to {final_end_date}")

    # Validate dates against final data range (COMPARISON SHOULD NOW WORK)
    try:
        # This comparison should now work correctly as all timestamps have the same awareness
        if val_start_date < final_start_date or test_start_date < val_start_date or end_date < test_start_date or test_start_date > final_end_date:
             print("Error: Date ranges invalid after feature engineering dropna.")
             print(f"  Available data range: {final_start_date} to {final_end_date}")
             print(f"  Configured Val Start: {val_start_date}")
             print(f"  Configured Test Start: {test_start_date}")
             print(f"  Configured End Date: {end_date}")
             # Provide more specific reasons
             if val_start_date < final_start_date: print("  Reason: Validation start date is earlier than the first available data point after cleaning/feature engineering.")
             if test_start_date < val_start_date: print("  Reason: Test start date is earlier than validation start date (logic error).") # Should have been caught earlier
             if end_date < test_start_date: print("  Reason: End date is earlier than test start date (logic error).") # Should have been caught earlier
             if test_start_date > final_end_date: print("  Reason: Test start date is later than the last available data point.")

             return None, None, {}, {}
    except TypeError as e:
        # This should ideally not happen after the sync, but catch just in case
        print(f"FATAL ERROR: Still encountered TypeError during date comparison after sync attempt: {e}")
        print(f"  val_start_date: {val_start_date} (tz={val_start_date.tzinfo})")
        print(f"  final_start_date: {final_start_date} (tz={final_start_date.tzinfo})")
        print(f"  test_start_date: {test_start_date} (tz={test_start_date.tzinfo})")
        print(f"  final_end_date: {final_end_date} (tz={final_end_date.tzinfo})")
        return None, None, {}, {}


    # --- Define Feature Columns (X) ---
    lag_cols = [col for col in df.columns if f"_lag_{lag_value}" in col]
    time_cols = []
    if add_time: time_cols = [col for col in df.columns if col not in original_cols and col not in y_curve_names and f"_lag_{lag_value}" not in col and "roll" not in col]
    rolling_cols = [col for col in df.columns if "roll" in col]
    X_cols = [];
    if add_lag: X_cols.extend(lag_cols)
    if add_rolling: X_cols.extend(rolling_cols)
    if add_time: X_cols.extend(time_cols)
    if not X_cols: print("Warning: No engineered features selected (add_lag/add_rolling/add_time are False). Using original X curves."); X_cols = [col for col in X_curve_names if col in df.columns]
    elif include_y_in_X: print("Including target variables in X."); targets_in_df = [col for col in y_curve_names if col in df.columns]; X_cols.extend(targets_in_df); X_cols = sorted(list(set(X_cols)))
    y_cols = [col for col in y_curve_names if col in df.columns]
    if not y_cols: print(f"Error: Target columns {y_curve_names} not in DataFrame after processing."); return None, None, {}, {}
    if len(y_cols) < len(y_curve_names): print(f"Warning: Missing targets in final DataFrame: {set(y_curve_names) - set(y_cols)}")
    X_cols = [col for col in X_cols if col in df.columns] # Ensure all selected X cols actually exist
    if not X_cols: print("Error: No feature columns (X_cols) found in DataFrame after processing."); return None, None, {}, {}
    print(f"Final feature columns (X): {X_cols}")
    print(f"Final target columns (y): {y_cols}")

    # --- Calculate Rolling CV Parameters from Dates ---
    X_df = df[X_cols]
    y_df = df[y_cols]
    total_len = len(X_df)

    # Find index location for val_start_date (this marks the end of the *first* training set)
    try:
        # searchsorted works correctly with tz-aware DatetimeIndex and tz-aware Timestamp
        first_train_end_idx = df.index.searchsorted(val_start_date, side='left')
        if first_train_end_idx == 0:
             # This check needs to be robust - compare against the *actual* first timestamp
             actual_first_ts = df.index[0]
             raise ValueError(f"val_start_date ({val_start_date}) is at or before the first data point "
                              f"after feature engineering ({actual_first_ts}). No training data available "
                              f"before validation period starts.")
        # Check if val_start_date actually exists in the index for clarity
        if first_train_end_idx < len(df.index) and df.index[first_train_end_idx] == val_start_date:
            print(f"First training set ends *before* index {first_train_end_idx} (timestamp {df.index[first_train_end_idx]})")
        elif first_train_end_idx < len(df.index):
             print(f"First training set ends at index {first_train_end_idx}, *before* the first timestamp >= val_start_date ({df.index[first_train_end_idx]})")
        else:
             print(f"First training set ends at index {first_train_end_idx} (end of data)")

    except ValueError as e: # Catch the specific error raised above
        print(f"Error determining initial training set end index: {e}")
        return None, None, {}, {}
    except Exception as e:
        print(f"Unexpected error determining index for val_start_date: {e}")
        return None, None, {}, {}

    # Function to find the round range for a given period
    def calculate_round_range(period_start_date, period_end_date, first_train_len, horizon, index):
        """Calculates the start (inclusive) and end (exclusive) round indices for a date period."""
        start_round = -1
        end_round = -1
        total_pts = len(index)
        # Estimate max rounds needed - can be large, maybe optimize later if needed
        max_possible_rounds = (total_pts - first_train_len) // horizon + 2 # Check a bit beyond theoretical max

        # Find first round where test window starts >= period_start_date
        for i in range(max_possible_rounds):
            test_start_idx = first_train_len + i * horizon
            if test_start_idx >= total_pts: break # No more data points left

            test_start_ts = index[test_start_idx]

            # Check if the timestamp at the start index is within the period
            if test_start_ts >= period_start_date:
                # Check if the *end* of this window is still valid (has enough points)
                test_end_idx = test_start_idx + horizon
                if test_end_idx > total_pts: continue # This round cannot be completed fully

                start_round = i
                break
        # If no round starts within or after the period start date
        if start_round == -1:
            print(f"Warning: No CV rounds found starting on or after {period_start_date}.")
            return 0, 0

        # Find first round where test window starts >= period_end_date (marks the end)
        end_round = -1 # Reset end_round marker
        for i in range(start_round, max_possible_rounds):
             test_start_idx = first_train_len + i * horizon
             test_end_idx = test_start_idx + horizon

             # Stop if this round goes past the total data length
             if test_start_idx >= total_pts:
                  end_round = i # This round couldn't even start, so previous one was last
                  break

             test_start_ts = index[test_start_idx]

             # If the start of this round's test window is at or after the period end,
             # then the previous round (i-1) was the last one whose test window
             # *started* before the period end. So the range is up to (but not including) i.
             if test_start_ts >= period_end_date:
                  end_round = i
                  break
        # If loop finishes without finding a round starting after period_end_date
        # it means all remaining rounds fit within the period.
        if end_round == -1:
             # Calculate total possible rounds based on remaining points
             remaining_pts = total_pts - (first_train_len + start_round * horizon)
             num_full_rounds_left = remaining_pts // horizon
             end_round = start_round + num_full_rounds_left
             # If there's a partial window at the end, it doesn't count as a full round here
             # The create_cross_validation_sets loop handles stopping correctly.

        # Final check: ensure end_round is not smaller than start_round
        if end_round < start_round:
            print(f"Warning: Calculated end_round ({end_round}) < start_round ({start_round}). Setting end_round = start_round.")
            end_round = start_round

        return start_round, end_round # end_round is exclusive for range()

    # Calculate validation rounds
    val_start_round, val_end_round = calculate_round_range(
        val_start_date, test_start_date, first_train_end_idx, crossval_horizon, df.index
    )
    # Calculate test rounds
    test_start_round, test_end_round = calculate_round_range(
        test_start_date, end_date, first_train_end_idx, crossval_horizon, df.index
    )

    print("\n--- Calculated CV Round Ranges ---")
    print(f"Validation Period ({val_start_date} to {test_start_date}): Rounds {val_start_round} to {val_end_round - 1} (Exclusive End)")
    print(f"Test Period ({test_start_date} to {end_date}): Rounds {test_start_round} to {test_end_round - 1} (Exclusive End)")

    if val_start_round >= val_end_round and test_start_round >= test_end_round:
        print("Error: No valid rounds found for either validation or test period based on calculated ranges.")
        print(f"  Check data availability between {val_start_date} and {end_date} relative to the first training end index {first_train_end_idx} and horizon {crossval_horizon}.")
        return None, None, {}, {}
    elif val_start_round >= val_end_round:
         print("Warning: No valid rounds found for the validation period.")
    elif test_start_round >= test_end_round:
         print("Warning: No valid rounds found for the test period.")


    # Define output paths and checkpoint files for each phase
    val_output_path = os.path.join(output_base_path, sub_area, "validation_rounds")
    test_output_path = os.path.join(output_base_path, sub_area, "test_rounds")
    val_checkpoint_file = os.path.join(val_output_path, "checkpoint.txt")
    test_checkpoint_file = os.path.join(test_output_path, "checkpoint.txt")

    # Prepare parameter dictionaries for the CV function
    val_params = {
        'name': 'Validation',
        'start_round': val_start_round,
        'end_round': val_end_round, # Exclusive end
        'output_path': val_output_path,
        'checkpoint_file': val_checkpoint_file
    }
    test_params = {
        'name': 'Test',
        'start_round': test_start_round,
        'end_round': test_end_round, # Exclusive end
        'output_path': test_output_path,
        'checkpoint_file': test_checkpoint_file
    }

    # --- Run Two-Stage Rolling CV Set Creation ---
    try:
        processed_val_count, processed_test_count = create_cross_validation_sets_and_save(
            X_df=X_df,
            y_df=y_df,
            first_train_len=first_train_end_idx, # Pass the calculated length
            crossval_horizon=crossval_horizon,
            X_to_forecast=X_to_forecast,
            session=session,
            val_params=val_params,
            test_params=test_params,
            batch_size=batch_size
        )
        print(f"\nSuccessfully processed {processed_val_count} validation rounds and {processed_test_count} test rounds.")

    except Exception as e:
        import traceback
        print(f"\n--- ERROR DURING CV SET CREATION ---")
        print(f"An error occurred: {e}")
        print(traceback.format_exc()) # Print full traceback for debugging
        print(f"Processing stopped. Check logs and checkpoint files.")
        # Return the calculated info even on failure for debugging
        return None, None, val_params, test_params

    # Return column names and the calculated round info for reference
    return X_cols, y_cols, val_params, test_params

# --- _get_data function ---
def _get_data(curve_names: list, target_columns: list,
              session: volue_insight_timeseries.Session,
              start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """ Retrieves data, resamples hourly, drops excessive NaNs (non-targets), drops rows with NaNs."""
    pandas_series = {}
    print("Fetching curves:")
    for curve_name in curve_names:
        print(f"  - {curve_name}")
        try:
            curve = session.get_curve(name=curve_name)
            if curve is None: print(f"Warning: Curve '{curve_name}' not found. Skipping."); continue
            ts = curve.get_data(data_from=start_date, data_to=end_date)
            if ts is None: print(f"Warning: No data for curve '{curve_name}'. Skipping."); continue
            s = ts.to_pandas()
            if s.empty or s.isnull().all(): print(f"Warning: Data for curve '{curve_name}' empty/all NaN. Skipping."); continue
            pandas_series[curve_name] = s
        except Exception as e: print(f"Error fetching '{curve_name}': {e}. Skipping.")
    if not pandas_series: print("Error: No data fetched."); return pd.DataFrame()
    print("Resampling hourly curves to 15min...")
    for col in pandas_series:
        if " h " in col or pandas_series[col].index.freq == 'H':
             print(f"  - Resampling {col}")
             pandas_series[col].index = pd.to_datetime(pandas_series[col].index)
             # Use reindex for robust filling across the full range
             full_range = pd.date_range(start=pandas_series[col].index.min(), end=pandas_series[col].index.max(), freq='15min')
             pandas_series[col] = pandas_series[col].reindex(full_range, method='ffill')
    print("Combining fetched data...")
    try:
        # Combine potentially sparse series carefully
        combined_df = pd.DataFrame(pandas_series)
        print(f"Combined DataFrame shape before cleaning: {combined_df.shape}")
    except Exception as e:
        print(f"Error combining Series: {e}"); df = pd.DataFrame()
        for name, s in pandas_series.items():
            try: df = pd.concat([df, s.rename(name)], axis=1)
            except Exception as concat_err: print(f"  - Error adding series {name}: {concat_err}")
        combined_df = df;
        if combined_df.empty: return combined_df
    # Drop non-target columns with high NaN percentage
    threshold = 0.4
    non_target_columns = [col for col in combined_df.columns if col not in target_columns]
    if non_target_columns:
        nan_percentages = combined_df[non_target_columns].isna().mean()
        cols_to_drop = nan_percentages[nan_percentages > threshold].index.tolist()
        if cols_to_drop: print(f"Dropping non-target columns with >{threshold*100}% NaN: {cols_to_drop}"); combined_df = combined_df.drop(columns=cols_to_drop)
    else: print("No non-target columns to check for NaNs.")
    # Drop rows with any remaining NaNs
    initial_rows = len(combined_df)
    cleaned_df = combined_df.dropna()
    rows_dropped = initial_rows - len(cleaned_df)
    print(f"Dropped {rows_dropped} rows containing NaN values.")
    if cleaned_df.empty: print("Warning: DataFrame is empty after dropping NaN rows.")
    return cleaned_df


def create_cross_validation_sets_and_save(X_df, y_df, first_train_len, crossval_horizon,
                                          X_to_forecast, session,
                                          val_params, test_params, # Dicts with phase info
                                          batch_size=10):
    """
    Creates rolling-origin CV sets for validation and test phases defined by round ranges.
    Saves files to phase-specific directories and uses phase-specific checkpoints.
    Resets file numbering when transitioning from validation to test phase.
    If a forecast can't be found (even after lookback), skips the round and continues.

    Parameters:
    -----------
    X_df, y_df: DataFrames with features and targets.
    first_train_len: Index marking the end of the initial training set (before val starts).
    crossval_horizon: Number of steps per round.
    X_to_forecast: Dict mapping columns to external forecast curve names.
    session: API session.
    val_params: Dict {'name', 'start_round', 'end_round', 'output_path', 'checkpoint_file'}.
    test_params: Dict {'name', 'start_round', 'end_round', 'output_path', 'checkpoint_file'}.
    batch_size: Rounds per memory cleanup batch.

    Returns:
        tuple: (processed_val_count, processed_test_count)
    """
    processed_val_count = 0
    processed_test_count = 0
    total_len = len(X_df)

    # Process validation rounds first, then test rounds, to ensure proper order
    # and reset file numbering between phases
    phases = [
        ('Validation', val_params, 0),  # Start val file numbering at 0
        ('Test', test_params, 0)        # Start test file numbering at 0
    ]

    for phase_name, params, starting_file_num in phases:
        # Skip this phase if no rounds to process
        if params['start_round'] >= params['end_round'] or params['start_round'] < 0:
            print(f"No {phase_name} rounds to process.")
            continue

        print(f"\n=== Starting {phase_name} Phase (Rounds {params['start_round']} to {params['end_round']-1}) ===")
        output_path = params['output_path']
        checkpoint_file = params['checkpoint_file']

        # --- Checkpoint Logic for this Phase ---
        resume_from_round = 0
        resume_from_file_num = starting_file_num
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = f.read().strip().split(',')
                    if len(checkpoint_data) >= 2 and checkpoint_data[0].isdigit() and checkpoint_data[1].isdigit():
                        # Format: "logical_round,file_number"
                        resume_from_round = int(checkpoint_data[0])
                        resume_from_file_num = int(checkpoint_data[1])
                        print(f"Resuming {phase_name} phase from logical round {resume_from_round} (file number {resume_from_file_num})")
                    elif len(checkpoint_data) == 1 and checkpoint_data[0].isdigit():
                        # Backward compatibility with old checkpoint format
                        resume_from_round = int(checkpoint_data[0])
                        resume_from_file_num = resume_from_round - params['start_round'] + starting_file_num
                        print(f"Resuming {phase_name} phase from round {resume_from_round} (old format)")
            except Exception as e:
                print(f"Warning: Could not read {phase_name} checkpoint '{checkpoint_file}': {e}")

        # Ensure output directory for this phase exists
        try:
            os.makedirs(output_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory '{output_path}': {e}")
            continue  # Skip this phase but try the next one

        # Process all rounds for this phase
        file_counter = resume_from_file_num  # Start file numbering from checkpoint or default
        for logical_round in range(max(params['start_round'], resume_from_round), params['end_round']):
            print(f"\n--- Processing {phase_name} Round {logical_round} (File #{file_counter}) ---")

            # Check if files already exist for this file counter in this phase's folder
            file_base_path = os.path.join(output_path, f"X_train_{file_counter}.csv")
            if os.path.exists(file_base_path):
                # Verify all four files exist before skipping
                files_exist = all(os.path.exists(os.path.join(output_path, f"{prefix}_{file_counter}.csv"))
                                 for prefix in ["X_train", "y_train", "X_test", "y_test"])
                if files_exist:
                    print(f"Files for {phase_name} round {logical_round} (File #{file_counter}) already exist, skipping...")
                    # Update checkpoint for this phase
                    try:
                        with open(checkpoint_file, 'w') as f:
                            f.write(f"{logical_round+1},{file_counter+1}")
                    except IOError as e:
                        print(f"Warning: Could not write {phase_name} checkpoint: {e}")
                    
                    # Increment counters
                    if phase_name == 'Validation':
                        processed_val_count += 1
                    else:
                        processed_test_count += 1
                    
                    file_counter += 1
                    continue
                else:
                    print(f"Warning: Found some but not all files for {phase_name} round {logical_round} (File #{file_counter}). Re-generating.")

            # Define indices
            train_end_idx = first_train_len + logical_round * crossval_horizon
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + crossval_horizon, total_len)

            # Check if there's enough data for a full prediction horizon
            if test_end_idx - test_start_idx < crossval_horizon:
                print(f"{phase_name} round {logical_round}: Insufficient data points ({test_end_idx - test_start_idx}) for full horizon ({crossval_horizon}). Stopping generation.")
                try:
                    # Checkpoint should reflect the round that failed
                    with open(checkpoint_file, 'w') as f:
                        f.write(f"{logical_round},{file_counter}")
                except IOError as e:
                    print(f"Warning: Could not write {phase_name} checkpoint: {e}")
                
                # Stop processing this phase
                break

            print(f"{phase_name} round {logical_round}: Train indices [0, {train_end_idx}), Test indices [{test_start_idx}, {test_end_idx})")

            # Extract data slices
            X_train = X_df.iloc[:train_end_idx]
            y_train = y_df.iloc[:train_end_idx]
            y_test = y_df.iloc[test_start_idx:test_end_idx]
            forecast_dates = y_test.index

            # Double-check length
            if len(forecast_dates) != crossval_horizon:
                print(f"Error: {phase_name} round {logical_round}: Forecast dates length ({len(forecast_dates)}) != horizon ({crossval_horizon}). Check logic.")
                try:
                    with open(checkpoint_file, 'w') as f:
                        f.write(f"{logical_round},{file_counter}")
                except IOError as e:
                    print(f"Warning: Could not write {phase_name} checkpoint: {e}")
                break

            # Generate forecast features for X_test
            print(f"Generating forecast features for {phase_name} round {logical_round}...")
            try:
                X_test = get_forecast(
                    dates=forecast_dates, 
                    X_train=X_train,
                    X_to_forecast=X_to_forecast, 
                    session=session,
                    max_lookback_hours=24,     # Add lookback parameters
                    lookback_step_minutes=15
                )
                
                # Verify index alignment post-forecast
                if not X_test.index.equals(forecast_dates):
                    print(f"Warning: {phase_name} round {logical_round}: X_test index mismatch after get_forecast. Re-aligning.")
                    X_test = X_test.reindex(forecast_dates)
                
                # Verify no NaNs remain after potential re-alignment
                if X_test.isnull().any().any():
                    # This indicates a problem in get_forecast or the source data
                    raise ValueError(f"NaNs found in X_test for {phase_name} round {logical_round} after forecasting/alignment.")

            except (RuntimeError, ValueError) as e:
                # CHANGE: Instead of raising the error, catch and log it, then skip this round
                print(f"--- ERROR: Failed forecast features for {phase_name} Round {logical_round} ---")
                print(f"  Reason: {e}")
                print(f"  No valid forecast found even after 24-hour lookback. Skipping this round.")
                
                # Update the checkpoint to move past this round
                try:
                    with open(checkpoint_file, 'w') as f:
                        f.write(f"{logical_round+1},{file_counter}")  # Notice we increment logical_round but NOT file_counter
                except IOError as cp_e:
                    print(f"Warning: Could not write {phase_name} checkpoint: {cp_e}")
                
                # Continue to the next round without incrementing file_counter
                continue  # Skip the rest of this iteration

            # Save files using the file counter (not the logical round number)
            print(f"Saving sets for {phase_name} round {logical_round} as file #{file_counter} to {output_path}")
            X_train.to_csv(os.path.join(output_path, f"X_train_{file_counter}.csv"), index=True)
            y_train.to_csv(os.path.join(output_path, f"y_train_{file_counter}.csv"), index=True)
            X_test.to_csv(os.path.join(output_path, f"X_test_{file_counter}.csv"), index=True)
            y_test.to_csv(os.path.join(output_path, f"y_test_{file_counter}.csv"), index=True)

            # Update checkpoint with both logical round and file counter
            try:
                with open(checkpoint_file, 'w') as f:
                    f.write(f"{logical_round+1},{file_counter+1}")
            except IOError as e:
                print(f"Warning: Could not write {phase_name} checkpoint: {e}")

            # Increment appropriate counter
            if phase_name == 'Validation':
                processed_val_count += 1
            else:
                processed_test_count += 1
            
            # Increment file counter
            file_counter += 1

            # Memory cleanup logic
            if file_counter % batch_size == 0:
                print(f"--- {phase_name} round {logical_round}: Cleaning memory after batch ---")
                # Delete large objects from this iteration
                del X_train, y_train, X_test, y_test
                # Explicitly call garbage collector
                gc.collect()

    print(f"\n--- Finished CV Set Generation: {processed_val_count} validation and {processed_test_count} test rounds ---")
    return processed_val_count, processed_test_count

def get_forecast(dates, X_train, X_to_forecast, session, max_retries=3, retry_delay=10,
                 max_lookback_hours=24, lookback_step_minutes=15):
    """
    Gets forecast using external curves. For instance curves, walks backwards in time
    to find the most recent forecast if the initial attempt fails.
    SPECIAL HANDLING: For 'pro no1 hydro tot mwh/h cet h f', directly attempts the
                      nearest previous midnight instance.

    Parameters:
    -----------
    dates: DatetimeIndex for the forecast period
    X_train: Training data (context only - might be unused if only forecasts needed)
    X_to_forecast: Dict mapping columns to external curve names
    session: API session
    max_retries: Max retries for each curve fetch
    retry_delay: Seconds to wait between retries
    max_lookback_hours: Maximum hours to look back for finding valid forecasts (for standard instance curves)
    lookback_step_minutes: Minutes to step back in each iteration (for standard instance curves)

    Returns:
    --------
    DataFrame with forecasted values for the requested dates

    Raises:
    -------
    RuntimeError: If external forecast fetching fails after all retries and lookbacks
    ValueError: If input parameters are invalid
    """
    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError("Error: 'dates' must be a pandas DatetimeIndex.")
    if not dates.tz:
        # Ensure dates have timezone for correct midnight calculation if source data is tz-aware
        print("Warning: 'dates' input to get_forecast is timezone-naive. Assuming UTC or system local time for midnight calculation if needed.")
        # toodoo considr localizing dates here: dates = dates.tz_localize('YourExpectedTimezone')

    if any(v is None for v in X_to_forecast.values()):
        raise ValueError("Error: X_to_forecast maps columns to external curve names (strings). None values no longer supported.")

    print(f"Generating forecast from external curves for dates: {dates.min()} to {dates.max()}")
    forecast_start_date = dates.min()
    forecast_end_date = dates.max()
    forecast_columns = list(X_to_forecast.keys())
    X_forecast = pd.DataFrame(index=dates, columns=forecast_columns)

    # Calculate total lookback steps based on hours and step size (for standard handling)
    max_lookback_steps = int((max_lookback_hours * 60) / lookback_step_minutes)

    # Define the specific curve requiring special handling
    SPECIAL_CURVE_NAME = "pro no1 hydro tot mwh/h cet h f" 

    # Iterate through columns needing forecasts
    for (orig_col, forecast_source_curve) in X_to_forecast.items():
        print(f"Processing forecast for: {orig_col}")
        print(f"  -> Fetching external forecast from curve: {forecast_source_curve}")
        success = False
        last_exception = None

        # Retry loop for fetching
        for attempt in range(max_retries):
            ts = None # Reset timeseries object for each attempt
            try:
                # Get curve object
                curve = session.get_curve(name=forecast_source_curve)
                if curve is None:
                    print(f"  -> Warning: Curve {forecast_source_curve} not found (attempt {attempt+1}/{max_retries}).")
                    last_exception = ValueError(f"Curve {forecast_source_curve} not found.")
                    time.sleep(retry_delay)
                    continue

                # --- Handle Instance Curves ---
                if isinstance(curve, InstanceCurve):

                    # --- >>> START SPECIAL MIDNIGHT HANDLING <<< ---
                    if forecast_source_curve == SPECIAL_CURVE_NAME:
                        # Calculate the nearest midnight before the forecast start date
                        # Subtract a tiny amount before normalizing to handle cases where forecast_start_date is exactly midnight
                        target_midnight_issue = (forecast_start_date - pd.Timedelta(microseconds=1)).normalize()
                        print(f"  -> SPECIAL HANDLING for {forecast_source_curve}: Targeting nearest previous midnight instance: {target_midnight_issue}")

                        try:
                            ts = curve.get_instance(issue_date=target_midnight_issue,
                                                   data_from=forecast_start_date,
                                                   data_to=forecast_end_date)
                            if ts is not None and hasattr(ts, 'to_pandas'):
                                s_check = ts.to_pandas()
                                if not s_check.empty and not s_check.isnull().all():
                                    print(f"  -> Successfully found valid midnight instance at {target_midnight_issue}")
                                    # ts is now valid and will be processed later
                                else:
                                    print(f"  -> Midnight instance at {target_midnight_issue} has no valid data.")
                                    ts = None # Mark as failed for this attempt
                            else:
                                print(f"  -> No data returned for midnight instance at {target_midnight_issue}")
                                ts = None # Mark as failed for this attempt

                            if ts is None:
                                # Raise specific error if midnight fetch failed, helps distinguish during retry
                                raise ValueError(f"Required midnight forecast at {target_midnight_issue} not found or invalid.")

                        except Exception as midnight_err:
                            print(f"  -> Failed special midnight fetch for {forecast_source_curve}: {midnight_err}")
                            last_exception = midnight_err
                            ts = None # Ensure ts is None if special fetch failed
                        # If ts is still None here, the special fetch failed for this attempt.
                        # The code will proceed to the common error handling below.

                    # --- >>> END SPECIAL MIDNIGHT HANDLING <<< ---

                    # --- Standard Lookback Logic for OTHER Instance Curves ---
                    else:
                        initial_issue_date = forecast_start_date - pd.Timedelta(minutes=lookback_step_minutes)
                        instance_success = False
                        instance_last_error = None

                        # Try walking back in time to find a valid forecast
                        for lookback_step in range(max_lookback_steps):
                            issue_date_attempt = initial_issue_date - pd.Timedelta(minutes=lookback_step * lookback_step_minutes)
                            print(f"  -> Attempting instance issued around: {issue_date_attempt} (lookback step {lookback_step+1}/{max_lookback_steps})")

                            try:
                                temp_ts = curve.get_instance(issue_date=issue_date_attempt,
                                                        data_from=forecast_start_date,
                                                        data_to=forecast_end_date)
                                if temp_ts is not None and hasattr(temp_ts, 'to_pandas'):
                                    s = temp_ts.to_pandas()
                                    if not s.empty and not s.isnull().all():
                                        print(f"  -> Successfully found valid instance at {issue_date_attempt}")
                                        ts = temp_ts # Assign the successful timeseries
                                        instance_success = True
                                        break # Found a valid instance, exit lookback loop
                                    else:
                                        print(f"  -> Instance at {issue_date_attempt} has no valid data")
                                        # Continue lookback
                                else:
                                    print(f"  -> No valid data returned for instance at {issue_date_attempt}")
                                    instance_last_error = ValueError(f"No valid data for instance at {issue_date_attempt}")
                            except Exception as instance_err:
                                print(f"  -> Failed instance {issue_date_attempt}: {instance_err}")
                                instance_last_error = instance_err
                                # Potentially add a small delay here if hitting API limits during lookback

                        # If lookback finished without success, store the last error
                        if not instance_success:
                            print(f"  -> All {lookback_step+1} lookback attempts failed for instance curve.")
                            last_exception = instance_last_error if instance_last_error else ValueError("Lookback failed without specific error.")
                            # ts remains None

                # --- Handle regular timeseries curves ---
                else:
                    ts = curve.get_data(data_from=forecast_start_date, data_to=forecast_end_date)
                    # Check if get_data returned None immediately
                    if ts is None:
                       print(f"  -> get_data returned None for {forecast_source_curve}")
                       last_exception = ValueError(f"get_data returned None for {forecast_source_curve}")


                # --- Common Processing & Error Handling for this Attempt ---
                if ts is None:
                    # This block is reached if any fetch method (special midnight, lookback, get_data)
                    # resulted in ts being None for this attempt.
                    print(f"  -> Warn: No valid data obtained for {forecast_source_curve} (attempt {attempt+1}/{max_retries}).")
                    # Ensure last_exception is set if it wasn't already
                    if last_exception is None:
                         last_exception = ValueError(f"No data obtained for {forecast_source_curve}, reason unclear.")
                    # Proceed to retry delay and next attempt
                    time.sleep(retry_delay)
                    continue # Go to the next retry attempt

                # --- Process the successfully fetched timeseries (ts) ---
                s = ts.to_pandas()
                if s.empty:
                    print(f"  -> Warn: Empty series for {forecast_source_curve} after fetch (attempt {attempt+1}/{max_retries}).")
                    last_exception = ValueError(f"Empty series for {forecast_source_curve}.")
                    time.sleep(retry_delay)
                    continue # Go to the next retry attempt

                # Align index and fill gaps (crucial step)
                s = s.reindex(dates, method='ffill').bfill()

                # Check for failure to align (all NaNs after reindex)
                if s.isnull().all():
                    error_msg = f"Forecast data for {orig_col} from {forecast_source_curve} entirely NaN after reindex (no overlap with {dates.min()} - {dates.max()})."
                    print(f"  -> Error: {error_msg}")
                    last_exception = ValueError(error_msg) # Store specific error
                    time.sleep(retry_delay) # Allow retry even for this
                    continue # Go to the next retry attempt

                # If we got here, s is valid, aligned, and has data
                X_forecast[orig_col] = s

                # Resample if needed (e.g., hourly source to 15min target)
                # Check source curve name OR frequency of the fetched series before resampling
                source_freq = pd.infer_freq(s.index) # Try to infer freq if needed
                target_freq = pd.infer_freq(dates)
                # Basic check using name, refine if needed based on actual frequencies
                if (" h " in forecast_source_curve or source_freq == 'h' or source_freq == 'H') and target_freq != source_freq:
                    print(f"  -> Resampling hourly data for {orig_col} to {target_freq}.")
                    # Resample to target frequency, then reindex again to be safe
                    X_forecast[orig_col] = X_forecast[orig_col].resample(target_freq).ffill().reindex(dates, method='ffill').bfill()

                    # Re-check for all NaNs after resampling
                    if X_forecast[orig_col].isnull().all():
                        error_msg = f"Forecast data for {orig_col} from {forecast_source_curve} entirely NaN after resampling/reindex."
                        print(f"  -> Error: {error_msg}")
                        last_exception = ValueError(error_msg)
                        time.sleep(retry_delay)
                        continue # Go to the next retry attempt

                print(f"  -> External forecast OK for {orig_col}.")
                success = True
                last_exception = None # Reset last exception on success
                break  # Exit retry loop for this curve

            except Exception as e:
                # Catch any other unexpected errors during the attempt
                print(f"  -> Error processing {forecast_source_curve} (attempt {attempt+1}/{max_retries}): {str(e)}")
                last_exception = e # Store the exception
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)  # Wait before retrying


        # --- After all retries for a curve ---
        if not success:
            error_msg = f"Failed fetch/process external forecast for {orig_col} from {forecast_source_curve} after {max_retries} attempts."
            print(f"  -> FINAL FAILURE: {error_msg}")
            # Raise runtime error, including the last specific exception encountered if available
            if last_exception:
                raise RuntimeError(error_msg, last_exception) from last_exception
            else:
                raise RuntimeError(error_msg)

    # --- After processing all curves ---
    # Final check for any columns that might still be all NaN (shouldn't happen if logic above is correct)
    if X_forecast.isnull().all().any():
        nan_cols = X_forecast.columns[X_forecast.isnull().all()].tolist()
        print(f"WARNING: Columns {nan_cols} are entirely NaN after forecast generation. Check data sources and alignment logic.")

    print("Forecast generation complete (or errors raised).")
    return X_forecast
