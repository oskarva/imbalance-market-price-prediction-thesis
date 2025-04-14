"""
Script to collect data and create two stages of rolling-origin cross-validation
splits (validation and test) based on specified date periods.
"""
import os
import argparse
import pandas as pd
from data.data_collection import get_data # Assuming get_data is in data_collection.py
from data.curves import get_curve_dicts # Assuming this defines curve sets
import volue_insight_timeseries
import sys

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Collect data & create two-stage rolling CV splits (validation & test) based on dates.'
    )

    # --- Date Arguments ---
    parser.add_argument('--start-date', type=str, default="2021-01-01",
                        help='Overall start date for data fetching (YYYY-MM-DD). Default: 2021-01-01')
    parser.add_argument('--val-start-date', type=str, default="2023-12-16",
                        help='Start date defining the validation period (YYYY-MM-DD HH:MM:SS or YYYY-MM-DD). Required.')
    parser.add_argument('--test-start-date', type=str, default="2024-06-16",
                        help='Start date defining the test period (YYYY-MM-DD HH:MM:SS or YYYY-MM-DD). Required.')
    parser.add_argument('--end-date', type=str, default="2025-03-15",
                        help='Overall end date for data fetching; defines end of test period (YYYY-MM-DD HH:MM:SS or YYYY-MM-DD). Required.')

    # --- Rolling CV Arguments ---
    parser.add_argument('--cv-horizon', type=int, default=32,
                        help='Number of steps (15-min intervals) to predict in each CV round. Default: 32 (8 hours)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of rounds per memory cleanup batch. Default: 10')

    # --- Feature Engineering Arguments ---
    #parser.add_argument('--add-time', action='store_true', help='Add time-based features.')
    #parser.add_argument('--add-lag', action='store_true', help='Add lag features.')
    #parser.add_argument('--add-rolling', action='store_true', help='Add rolling window features.')
    parser.add_argument('--lag-value', type=int, default=32, help='Periods for lag features. Default: 32')
    #parser.add_argument('--include-y-in-x', action='store_true', help='Include target variables (y) in the feature set (X).')

    # --- Path Arguments ---
    parser.add_argument('--output-dir', type=str, default="./src/data/csv",
                        help='Base directory to save the output folders (validation_rounds, test_rounds). Default: ./data/two_stage_cv')

    # --- Other Arguments ---
    parser.add_argument('--area', type=str, default="no", help='Area code for curve dictionaries. Default: "no".')

    args = parser.parse_args()

    # --- Validate and Process Args ---
    try:
        start_date = pd.Timestamp(args.start_date)
        val_start_date = pd.Timestamp(args.val_start_date)
        test_start_date = pd.Timestamp(args.test_start_date)
        end_date = pd.Timestamp(args.end_date)

        if not start_date < val_start_date < test_start_date < end_date:
             raise ValueError("Dates must be ordered: start_date < val_start_date < test_start_date < end_date")
        if args.cv_horizon <= 0: raise ValueError("--cv-horizon must be positive")
        if args.batch_size <= 0: raise ValueError("--batch-size must be positive")

    except Exception as e:
        print(f"Error parsing or validating arguments: {e}")
        sys.exit(1)

    # Print configuration details
    print("--- Data Collection Configuration (Two-Stage Rolling CV) ---")
    print(f"Overall Data Range: {start_date} to {end_date}")
    print(f"Validation Period Start: {val_start_date}")
    print(f"Test Period Start:       {test_start_date}")
    print(f"Test Period End:         {end_date}")
    print(f"CV Horizon:         {args.cv_horizon} steps ({args.cv_horizon*15} minutes)")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Area:               {args.area}")
    print(f"Output Directory:   {args.output_dir}")
    #print("\n--- Feature Engineering Options ---")
    #print(f"Add Time Features:  {args.add_time}")
    #print(f"Add Lag Features:   {args.add_lag} (Lag Value: {args.lag_value if args.add_lag else 'N/A'})")
    #print(f"Add Rolling Feat.:  {args.add_rolling}")
    #print(f"Include y in X:     {args.include_y_in_x}")
    print("---------------------------------------------------------\n")


    # --- Get Curve Definitions ---
    try:
        collections = get_curve_dicts(area=args.area)
        if not collections: print(f"Error: No curve collections for area '{args.area}'."); sys.exit(1)
    except Exception as e: print(f"Error getting curve dictionaries: {e}"); sys.exit(1)


    # --- Initialize API Session ---
    wapi_config_path = os.environ.get("WAPI_CONFIG")
    if not wapi_config_path: print("Error: WAPI_CONFIG env var not set."); sys.exit(1)
    try:
        session = volue_insight_timeseries.Session(config_file=wapi_config_path); print("API Session initialized.")
    except Exception as e: print(f"Error initializing API session: {e}"); sys.exit(1)


    # --- Process Each Curve Collection ---
    for collection in collections:
        sub_area = collection.get("sub_area", "unknown_sub_area")
        print(f"\n=== Processing Collection for Sub-Area: {sub_area} ===")
        X_curve_names = collection.get("X"); target_curves = collection.get("y")
        X_to_forecast = collection.get("X_to_forecast", {})

        #EXCEPTION: NO5 does not have wind data
        if sub_area == "no5":
            X_curve_names.remove("pro no5 wnd mwh/h cet min15 a")
            X_to_forecast.pop("pro no5 wnd mwh/h cet min15 a")
        # Basic validation of collection content
        if not X_curve_names or not target_curves: print(f"Warn: Skipping '{sub_area}', missing 'X'/'y'."); continue
        if not X_to_forecast: print(f"Warn: Skipping '{sub_area}', missing 'X_to_forecast'."); continue
        if any(v is None for v in X_to_forecast.values()):
            print(f"Error: Skipping '{sub_area}', 'X_to_forecast' contains None values (Prophet not supported).")
            continue

        # Call the main data processing and CV splitting function
        X_columns, y_columns, val_info, test_info = get_data(
            X_curve_names=X_curve_names, y_curve_names=target_curves,
            sub_area=sub_area, session=session,
            start_date=start_date,
            val_start_date=val_start_date, # Pass dates defining periods
            test_start_date=test_start_date,
            end_date=end_date,
            output_base_path=args.output_dir,
            X_to_forecast=X_to_forecast,
            crossval_horizon=args.cv_horizon,
            batch_size=args.batch_size,
            add_time=False, add_lag=False, add_rolling=False,
            include_y_in_X=False, lag_value=args.lag_value
        )

        # Report results for the collection
        if X_columns and y_columns:
            print(f"\nTwo-Stage CV data generation process finished for {sub_area}.")
            print(f"  Validation Rounds Info: Start={val_info.get('start_round', 'N/A')}, End={val_info.get('end_round', 'N/A')}")
            print(f"  Test Rounds Info: Start={test_info.get('start_round', 'N/A')}, End={test_info.get('end_round', 'N/A')}")
            print(f"  Feature Columns: {len(X_columns)}")
            print(f"  Target Columns:  {len(y_columns)}")
            print(f"  Validation data in: {val_info.get('output_path', 'N/A')}")
            print(f"  Test data in: {test_info.get('output_path', 'N/A')}")
        else:
            print(f"\nTwo-Stage CV data generation failed for {sub_area}. Check logs.")
        print(f"=== Finished Collection for Sub-Area: {sub_area} ===")

    print("\nAll collections processed.")

if __name__ == "__main__":
    main()
