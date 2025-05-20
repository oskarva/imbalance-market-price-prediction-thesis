"""
Evaluate forecast accuracy across areas and test rounds.
Generates a CSV summary with MAE, RMSE, and R2 for each variable per area.
"""
import os
import sys
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.data.curves import get_curve_dicts

def main():
    areas = ['no1', 'no2', 'no3', 'no4', 'no5']
    results = []
    # Iterate through each area
    for area in areas:
        base_dir = f'/Volumes/T9/{area}/test_rounds'
        if not os.path.isdir(base_dir):
            print(f'Warning: directory not found: {base_dir}', file=sys.stderr)
            continue
        # Get mapping of actual to forecast column names for this sub-area
        # get_curve_dicts expects main area 'no', sub_areas=[area]
        curves = get_curve_dicts('no', sub_areas=[area])[0]
        mapping = curves['X_to_forecast']  # dict: actual_col -> forecast_col

        # Accumulate metrics per variable
        var_acc = {act: [] for act in mapping.keys()}
        # Gather test files and identify the final training file
        test_files = [f for f in os.listdir(base_dir) if re.match(r'X_test_\d+\.csv$', f)]
        if not test_files:
            continue
        test_files = sorted(test_files)
        # Determine final training file (max round)
        rounds = [int(re.match(r'X_test_(\d+)\.csv$', f).group(1)) for f in test_files]
        max_N = max(rounds)
        # Load final actuals once: use X_train_{max_N+1} if exists else X_train_{max_N}
        act_file = f'X_train_{max_N+1}.csv'
        act_path = os.path.join(base_dir, act_file)
        if not os.path.exists(act_path):
            act_file = f'X_train_{max_N}.csv'
            act_path = os.path.join(base_dir, act_file)
        if not os.path.exists(act_path):
            print(f'No actuals file for area {area}', file=sys.stderr)
            continue
        df_act = pd.read_csv(act_path, index_col=0, parse_dates=True)
        df_act.index = pd.to_datetime(df_act.index, utc=True)
        # Process each test file
        for fname in test_files:
            # skip final round if no later actuals
            N = int(re.match(r'X_test_(\d+)\.csv$', fname).group(1))
            if N == max_N:
                continue
            test_path = os.path.join(base_dir, fname)
            # Read test forecasts and normalize to UTC
            df_test = pd.read_csv(test_path, index_col=0, parse_dates=True)
            df_test.index = pd.to_datetime(df_test.index, utc=True)
            # Align timestamps with final actuals
            common_idx = df_test.index.intersection(df_act.index)
            # Evaluate each variable
            for act_col, fcol in mapping.items():
                # ensure actual column exists
                if act_col not in df_act.columns:
                    continue
                # pick forecast column: mapping value or fallback to actual column name
                if fcol in df_test.columns:
                    pred_col = fcol
                elif act_col in df_test.columns:
                    pred_col = act_col
                else:
                    continue
                y_true = df_act.loc[common_idx, act_col]
                y_pred = df_test.loc[common_idx, pred_col]
                mask = y_true.notna() & y_pred.notna()
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                if len(y_true) == 0:
                    continue
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                var_acc[act_col].append((mae, rmse, r2))
        # Aggregate metrics per variable
        for act_col, metrics in var_acc.items():
            if not metrics:
                continue
            maes, rmses, r2s = zip(*metrics)
            results.append({
                'area': area,
                'variable': act_col,
                'MAE': np.mean(maes),
                'RMSE': np.mean(rmses),
                'R2': np.mean(r2s),
                'n_instances': len(metrics)
            })
    # Save summary
    df_res = pd.DataFrame(results)
    # Save per-area, per-feature metrics
    out_file = 'forecast_accuracy_summary.csv'
    df_res.to_csv(out_file, index=False)
    print(f'Wrote forecast accuracy summary to {out_file}')
    # Aggregate metrics across areas for each feature (strip area code)
    df_res['feature'] = (
        df_res['variable']
        .str.replace(r'\sno[1-5]\s', ' ', regex=True)
        .str.strip()
    )
    df_overall = (
        df_res
        .groupby('feature')[['MAE','RMSE','R2']]
        .mean()
        .reset_index()
        .rename(columns={'feature': 'variable'})
    )
    out_overall = 'forecast_accuracy_summary_by_variable.csv'
    df_overall.to_csv(out_overall, index=False)
    print(f'Wrote averaged metrics by variable to {out_overall}')

if __name__ == '__main__':
    main()