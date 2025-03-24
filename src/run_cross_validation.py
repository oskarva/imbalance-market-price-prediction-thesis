"""
Main script to run cross-validation with memory efficiency.
"""
import os
import argparse
import time
from models.cross_validation_training import run_cross_validation
import json

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run cross-validation for time series forecasting')
    
    parser.add_argument('--start', type=int, default=0,
                        help='First CV round to process (default: 0)')
    
    parser.add_argument('--end', type=int, default=None,
                        help='Last CV round to process (default: all available)')
    
    parser.add_argument('--step', type=int, default=1,
                        help='Step size for processing rounds (default: 1)')
    
    parser.add_argument('--output', type=str, default='./results/cv_run',
                        help='Directory to save results (default: ./results/cv_run)')
    
    parser.add_argument('--n_estimators', type=int, default=1000,
                        help='Number of estimators for XGBoost (default: 1000)')
    
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='Learning rate for XGBoost (default: 0.05)')
    
    parser.add_argument('--max_depth', type=int, default=20,
                        help='Max depth for XGBoost (default: 20)')
    
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='Subsample ratio for XGBoost (default: 1.0)')
    
    parser.add_argument('--colsample_bytree', type=float, default=1.0,
                        help='Column subsample ratio for XGBoost (default: 1.0)')
    
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save results (default: False)')
    
    args = parser.parse_args()
    
    # Prepare output directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"{args.output}_{timestamp}"
    
    if not args.no_save:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save run configuration
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Configure XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'random_state': 42
    }
    
    print(f"Starting cross-validation run with the following configuration:")
    print(f"  Rounds: {args.start} to {args.end if args.end else 'end'} with step {args.step}")
    print(f"  Output directory: {output_dir}")
    print(f"  XGBoost parameters: {xgb_params}")
    
    # Run cross-validation
    start_time = time.time()
    
    results = run_cross_validation(
        params=xgb_params,
        start_round=args.start,
        end_round=args.end,
        step=args.step,
        save_results=not args.no_save,
        output_dir=output_dir
    )
    
    total_time = time.time() - start_time
    
    print(f"\nCross-validation completed in {total_time:.2f} seconds")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
