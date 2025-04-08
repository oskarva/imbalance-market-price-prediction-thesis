"""
Script to resume data collection from a specific round.
"""
import os
import argparse
import pandas as pd
from data.data_collection import get_data
from data.curves import get_curve_dicts
import volue_insight_timeseries

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Resume data collection from a specific round')
    
    parser.add_argument('--start', type=int, default=0,
                        help='Round number to start from (e.g., 83)')
    
    parser.add_argument('--end', type=int, default=None,
                        help='Round number to end at (default: process all remaining rounds)')
    
    parser.add_argument('--batch', type=int, default=5,
                        help='Batch size for memory cleanup (default: 5)')
    
    parser.add_argument('--start-date', type=str, default="2021-01-01",
                        help='Start date for data collection (default: 2021-01-01)')
    
    parser.add_argument('--end-date', type=str, default="2025-03-18",
                        help='End date for data collection (default: 2025-03-18)')
    
    args = parser.parse_args()
    
    # Set up dates
    start_date = pd.Timestamp(args.start_date)
    if args.end_date:
        end_date = pd.Timestamp(args.end_date)
    else:
        end_date = pd.Timestamp.now()
    
    print(f"Resuming data collection from round {args.start}")
    print(f"Data range: {start_date} to {end_date}")
    print(f"Batch size: {args.batch}")
    
    collections = get_curve_dicts(area="no")
    for collection in collections:
        X_curve_names = collection["X"]
        target_curves = collection["y"]
        X_to_forecast = collection["X_to_forecast"]

        # Initialize the API session
        session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

        # Call the data collection function with the start round parameter
        X, y, X_columns, y_columns, n_rounds = get_data(
            X_curve_names, target_curves, collection["sub_area"],
            session,
            start_date, end_date,
            X_to_forecast,
            add_time=False, 
            add_lag=False,
            add_rolling=False,
            batch_size=args.batch,
            start_round=args.start,
            end_round=args.end
        )

        print(f"Data collection completed for {collection['sub_area']}. {n_rounds} total rounds available.")

if __name__ == "__main__":
    main()