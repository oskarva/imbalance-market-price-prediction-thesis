"""
Compare feature distributions when spot price matches regulation vs deviates.
"""
import os
import re
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data.curves import get_curve_dicts

def descriptive_stats(df, features):
    stats = []
    for feat in features:
        s = df[feat]
        stats.append({
            'feature': feat,
            'mean': s.mean(),
            'median': s.median(),
            'std': s.std(),
            '10%': s.quantile(0.1),
            '90%': s.quantile(0.9)
        })
    return pd.DataFrame(stats)

def main():
    # Run comparison for all areas
    for area in ["no1", "no2", "no3", "no4", "no5"]:
        loop(area)

def loop(area, output='./results/compare_plots'):
    """
    For a given area, load the latest X_train and y_train datasets and
    compare feature distributions when spot price matches vs deviates
    from regulation targets.
    """
    # Prepare output directory for this area
    out_dir = os.path.join(output, area)
    os.makedirs(out_dir, exist_ok=True)
    # Base directory for the CSV files
    base_dir = f'/Volumes/T9/{area}/test_rounds'
    # Identify latest X_train_N.csv and y_train_N.csv
    x_files = [f for f in os.listdir(base_dir) if re.match(r'X_train_(\d+)\.csv$', f)]
    y_files = [f for f in os.listdir(base_dir) if re.match(r'y_train_(\d+)\.csv$', f)]
    if not x_files or not y_files:
        print(f'No train files for area {area}', file=sys.stderr)
        return
    # Pick max N for each
    xN = max(int(re.match(r'X_train_(\d+)\.csv$', f).group(1)) for f in x_files)
    yN = max(int(re.match(r'y_train_(\d+)\.csv$', f).group(1)) for f in y_files)
    x_path = os.path.join(base_dir, f'X_train_{xN}.csv')
    y_path = os.path.join(base_dir, f'y_train_{yN}.csv')
    # Load feature and target data
    df_x = pd.read_csv(x_path, index_col=0, parse_dates=True)
    df_y = pd.read_csv(y_path, index_col=0, parse_dates=True)
    # Normalize indices to UTC and merge
    df_x.index = pd.to_datetime(df_x.index, utc=True)
    df_y.index = pd.to_datetime(df_y.index, utc=True)
    df = df_x.join(df_y, how='inner')
    # Get curve definitions (features, targets, spot mapping)
    curves = get_curve_dicts('no', sub_areas=[area])[0]
    # Identify expected features and filter to those present
    all_features = curves['X']
    features = [f for f in all_features if f in df.columns]
    missing = set(all_features) - set(features)
    if missing:
        print(f'Warning: missing features for area {area}: {missing}', file=sys.stderr)
    target_up = curves['y'][0]
    target_down = curves['y'][1]
    spot_key = next(k for k,v in curves['X_to_forecast'].items() if 'spot' in v.lower())
    spot = curves['X_to_forecast'][spot_key]

    # Helper to sanitize filenames
    def sanitize(fn):
        # replace any character not alnum, dot, underscore, or hyphen with underscore
        return re.sub(r'[^A-Za-z0-9_.-]', '_', fn)
    # For each direction (up/down), compare match vs deviation on filtered timestamps
    for direction, target in [('up', target_up), ('down', target_down)]:
        # Load filtered timestamps (naive) for this direction
        filt_path = os.path.join(
            './results/predictions/test_filtered',
            f'{area}_{direction}_naive_filtered.csv'
        )
        if not os.path.exists(filt_path):
            print(f'Filtered file missing: {filt_path}', file=sys.stderr)
            continue
        df_filt = pd.read_csv(filt_path, parse_dates=['timestamp'])
        df_filt['timestamp'] = pd.to_datetime(df_filt['timestamp'], utc=True)
        # Load all naive prediction timestamps for this direction
        pred_file = f'{area}_{direction}_naive_predictions.csv'
        pred_path = os.path.join('./results/predictions/test', pred_file)
        if not os.path.exists(pred_path):
            print(f'Predictions file missing: {pred_path}', file=sys.stderr)
            continue
        df_preds = pd.read_csv(pred_path, parse_dates=['timestamp'])
        df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'], utc=True)
        # Compute deviation and match timestamp sets
        dev_set = set(df_filt['timestamp'])
        all_ts = df_preds['timestamp']
        match_ts = [t for t in all_ts if t not in dev_set]
        # Subset main dataframe on match/deviation timestamps
        idx_dev = df.index.intersection(df_filt['timestamp'])
        idx_match = df.index.intersection(match_ts)
        df_dev = df.loc[idx_dev]
        df_match = df.loc[idx_match]
        # Descriptive stats
        stats_match = descriptive_stats(df_match, features)
        stats_dev   = descriptive_stats(df_dev, features)
        summary = stats_match.merge(
            stats_dev, on='feature', suffixes=(f'_{direction}_match', f'_{direction}_dev')
        )
        out_stats = os.path.join(out_dir, f'feature_stats_{direction}_match_vs_dev.csv')
        summary.to_csv(out_stats, index=False)
        print(f'Wrote stats for {direction} to', out_stats)
        # Plot distributions per feature
        for feat in features:
            # Density plots
            plt.figure(figsize=(8,4))
            sns.kdeplot(df_match[feat].dropna(), label=f'{direction.upper()} Match', bw_adjust=1)
            sns.kdeplot(df_dev[feat].dropna(),   label=f'{direction.upper()} Deviation', bw_adjust=1)
            plt.title(f'{direction.upper()} Density: {feat}')
            plt.legend()
            plt.tight_layout()
            fn_density = sanitize(f'{direction}_{feat}_density.png')
            plt.savefig(os.path.join(out_dir, fn_density))
            plt.close()
            # Boxplots
            plt.figure(figsize=(6,4))
            sns.boxplot(
                data=[df_match[feat].dropna(), df_dev[feat].dropna()],
                palette=['C0','C1']
            )
            plt.xticks([0,1], ['Match','Deviation'])
            plt.title(f'{direction.upper()} Boxplot: {feat}')
            plt.tight_layout()
            fn_box = sanitize(f'{direction}_{feat}_boxplot.png')
            plt.savefig(os.path.join(out_dir, fn_box))
            plt.close()

if __name__ == '__main__':
    main()