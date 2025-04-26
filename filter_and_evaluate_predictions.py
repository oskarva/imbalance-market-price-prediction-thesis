"""
filter_and_evaluate_predictions.py

Loads the spot==target matches pickle and prediction CSVs, filters out matching timestamps,
and computes R², MAE, RMSE on the remaining points. Saves filtered prediction files to a
separate folder and writes a summary CSV of metrics.
"""
import os
import argparse
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter out spot==target predictions and recompute metrics"
    )
    parser.add_argument(
        "--pred-dir", type=str,
        default=os.path.join("results", "predictions", "test"),
        help="Directory containing original prediction CSVs (area_direction[_model]_predictions.csv)"
    )
    parser.add_argument(
        "--matches-pkl", type=str,
        default="exact_spot_matches.pkl",
        help="Pickle file containing timestamps where target == spot per area/target"
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=os.path.join("results", "predictions", "test_filtered"),
        help="Directory to write filtered prediction CSVs and metrics summary"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Load spot==target timestamps
    with open(args.matches_pkl, "rb") as f:
        matches = pickle.load(f)

    # Prepare output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Define model suffixes for file lookup: naive, xgb, and blank for stacked/ebm
    suffixes = ["_naive", "_xgb", ""]
    # Collect metrics
    metrics = []

    for area, targets in matches.items():
        for target_col, info in targets.items():
            # Determine direction from target column name
            key = target_col.lower()
            if "regulation up" in key:
                direction = "up"
            elif "regulation down" in key:
                direction = "down"
            else:
                print(f"Skipping unknown target '{target_col}' in area {area}")
                continue

            # Build set of timestamps to drop (convert to UTC)
            drop_ts = set()
            for ts in info["timestamps"]:
                # ensure tz-aware
                try:
                    ts_utc = ts.tz_convert('UTC')
                except Exception:
                    ts_utc = ts.tz_localize('UTC')
                drop_ts.add(ts_utc)

            # Read predictions
            for suffix in suffixes:
                fname = f"{area}_{direction}{suffix}_predictions.csv"
                in_path = os.path.join(args.pred_dir, fname)
                if not os.path.isfile(in_path):
                    continue
                df = pd.read_csv(in_path, parse_dates=['timestamp'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                # Filter out matching timestamps
                mask_keep = ~df['timestamp'].isin(drop_ts)
                df_filt = df[mask_keep].copy()
                orig_n = len(df)
                filt_n = len(df_filt)
                pct_removed = 100 * (1 - filt_n / orig_n) if orig_n else np.nan

                if suffix:  # naive or xgb
                    model_label = suffix.lstrip('_')
                    # use 'predicted' column
                    df_pred = df_filt[['actual', 'predicted', 'timestamp', 'round']]
                    # compute metrics
                    if df_pred.empty:
                        r2 = mae = rmse = np.nan
                    else:
                        y_true = df_pred['actual'].values
                        y_pred = df_pred['predicted'].values
                        r2 = r2_score(y_true, y_pred)
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    # record
                    metrics.append({
                        'area': area,
                        'direction': direction,
                        'model': model_label,
                        'n_original': orig_n,
                        'n_filtered': filt_n,
                        'pct_removed': pct_removed,
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                    })
                    # save filtered file
                    out_fname = f"{area}_{direction}_{model_label}_filtered.csv"
                    df_pred.to_csv(os.path.join(args.out_dir, out_fname), index=False)
                    print(f"Saved filtered predictions to {out_fname}")
                else:
                    # stacked: first EBM alone, then combined
                    # EBM only
                    if 'ebm_pred' in df_filt.columns:
                        df_ebm = df_filt[['actual', 'ebm_pred', 'timestamp', 'round']].rename(
                            columns={'ebm_pred': 'predicted'})
                        if df_ebm.empty:
                            r2 = mae = rmse = np.nan
                        else:
                            y_true = df_ebm['actual'].values
                            y_pred = df_ebm['predicted'].values
                            r2 = r2_score(y_true, y_pred)
                            mae = mean_absolute_error(y_true, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        metrics.append({
                            'area': area,
                            'direction': direction,
                            'model': 'ebm',
                            'n_original': orig_n,
                            'n_filtered': filt_n,
                            'pct_removed': pct_removed,
                            'r2': r2,
                            'mae': mae,
                            'rmse': rmse,
                        })
                        out_fname = f"{area}_{direction}_ebm_filtered.csv"
                        df_ebm.to_csv(os.path.join(args.out_dir, out_fname), index=False)
                        print(f"Saved filtered EBM predictions to {out_fname}")
                    # Combined stacked predictions
                    df_stack = df_filt[['actual', 'predicted', 'timestamp', 'round']]
                    if df_stack.empty:
                        r2 = mae = rmse = np.nan
                    else:
                        y_true = df_stack['actual'].values
                        y_pred = df_stack['predicted'].values
                        r2 = r2_score(y_true, y_pred)
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    metrics.append({
                        'area': area,
                        'direction': direction,
                        'model': 'stacked',
                        'n_original': orig_n,
                        'n_filtered': filt_n,
                        'pct_removed': pct_removed,
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                    })
                    out_fname = f"{area}_{direction}_stacked_filtered.csv"
                    df_stack.to_csv(os.path.join(args.out_dir, out_fname), index=False)
                    print(f"Saved filtered stacked predictions to {out_fname}")

    # Save metrics summary
    metrics_df = pd.DataFrame(metrics)
    metrics_out = os.path.join(args.out_dir, 'filtered_metrics_summary.csv')
    metrics_df.to_csv(metrics_out, index=False)
    print(f"Wrote metrics summary to {metrics_out}")


if __name__ == '__main__':
    main()