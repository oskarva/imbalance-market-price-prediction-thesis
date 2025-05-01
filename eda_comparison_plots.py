import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import glob
import re

# --- Data loading utilities (for test_rounds) ---
def find_last_round_files(folder):
    """Finds the X_train and y_train files with the highest round number in 'folder'."""
    x_pattern = os.path.join(folder, 'X_train_*.csv')
    y_pattern = os.path.join(folder, 'y_train_*.csv')
    x_files = glob.glob(x_pattern)
    y_files = glob.glob(y_pattern)
    if not x_files or not y_files:
        return None, None, -1
    max_n = -1
    for f in x_files:
        base = os.path.basename(f)
        m = re.match(r'X_train_(\d+)\.csv$', base)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    if max_n < 0:
        return None, None, -1
    x_file = os.path.join(folder, f'X_train_{max_n}.csv')
    y_file = os.path.join(folder, f'y_train_{max_n}.csv')
    if not os.path.exists(x_file) or not os.path.exists(y_file):
        return None, None, max_n
    return x_file, y_file, max_n

def load_and_prepare_data(x_path, y_path):
    """Load X and y CSVs, parse datetime index, and merge into one DataFrame."""
    try:
        df_x = pd.read_csv(x_path, index_col=0)
        df_y = pd.read_csv(y_path, index_col=0)
    except Exception as e:
        print(f"[Error] Could not read CSVs: {e}")
        return None
    # Parse timestamps with UTC and convert to Europe/Oslo if possible
    df_x.index = pd.to_datetime(df_x.index, utc=True, errors='coerce')
    df_y.index = pd.to_datetime(df_y.index, utc=True, errors='coerce')
    df_x = df_x[df_x.index.notna()]
    df_y = df_y[df_y.index.notna()]
    if df_x.empty or df_y.empty:
        return None
    # Inner join on timestamp
    df = df_x.join(df_y, how='inner')
    try:
        df.index = df.index.tz_convert('Europe/Oslo')
    except Exception:
        pass
    return df

def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def sanitize_filename(s):
    """Sanitize string to be safe as a filename."""
    # Replace non-alphanumeric characters with underscore
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')

def plot_within_zone_deviation(df, zone, tolerance, output_dir):
    """
    Generate comparative plots for key features within a single zone,
    showing distributions by deviation state: Match, Up Deviation, Down Deviation.
    """
    # Define column names
    target_up = f"pri {zone} regulation up €/mwh cet min15 a"
    target_down = f"pri {zone} regulation down €/mwh cet min15 a"
    spot_price = f"pri {zone} spot €/mwh cet h a"
    features = [
        f"rdl {zone} mwh/h cet min15 a",
        f"con {zone} intraday mwh/h cet h a",
        f"pro {zone} hydro tot mwh/h cet h af"
    ]
    # Compute masks for categories
    mask_up_match   = np.isclose(df[target_up],   df[spot_price], atol=tolerance)
    mask_down_match = np.isclose(df[target_down], df[spot_price], atol=tolerance)
    mask_up_dev     = (df[target_up]   - df[spot_price])   > tolerance
    mask_down_dev   = (df[spot_price] - df[target_down]) > tolerance
    # Prepare output directory
    ensure_dir(output_dir)
    # For each feature, build a DataFrame with overlapping entries for independent categories
    for feat in features:
        if feat not in df.columns:
            print(f"[Warning] Feature '{feat}' not found for zone {zone}, skipping.")
            continue
        parts = []
        # Each category independently
        for label, mask in [
            ('Match Up',   mask_up_match),
            ('Match Down', mask_down_match),
            ('Up Deviation',   mask_up_dev),
            ('Down Deviation', mask_down_dev)
        ]:
            tmp = df.loc[mask, [feat]].dropna().copy()
            if tmp.empty:
                continue
            tmp['deviation_state'] = label
            parts.append(tmp)
        if not parts:
            print(f"[Info] No data for any category on feature '{feat}', skipping.")
            continue
        sub = pd.concat(parts, ignore_index=True)
        order = ['Match Up', 'Match Down', 'Up Deviation', 'Down Deviation']
        # Box plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=sub, x='deviation_state', y=feat, order=order, whis=(5, 95))
        plt.title(f"Distribution of {zone.upper()} {feat} by Deviation State")
        plt.xlabel('Deviation State')
        plt.ylabel(feat)
        fname = f"{zone.upper()}_{sanitize_filename(feat)}_by_State_boxplot.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        # KDE plot with dashed match lines
        plt.figure(figsize=(8, 6))
        ax = sns.kdeplot(
            data=sub, x=feat, hue='deviation_state', common_norm=False,
            hue_order=order
        )
        # Dashed style for match lines
        for line in ax.get_lines():
            lbl = line.get_label()
            if lbl in ['Match Up', 'Match Down']:
                line.set_linestyle('--')
        plt.title(f"Distribution of {zone.upper()} {feat} by Deviation State (KDE)")
        plt.xlabel(feat)
        plt.ylabel('Density')
        fname_kde = f"{zone.upper()}_{sanitize_filename(feat)}_by_State_kde.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname_kde))
        plt.close()
        # Additional boxplot: RDL outside deviations (match states only)
        if feat.lower().startswith('rdl'):
            # Keep only match states
            match_df = sub[sub['deviation_state'].isin(['Match Up', 'Match Down'])]
            if not match_df.empty:
                plt.figure(figsize=(6, 6))
                sns.boxplot(data=match_df, x='deviation_state', y=feat,
                            order=['Match Up', 'Match Down'], palette='Set2', whis=(5, 95))
                plt.title(f"Distribution of {zone.upper()} {feat} Outside Deviations")
                plt.xlabel('Match State')
                plt.ylabel(feat)
                fn = f"{zone.upper()}_{sanitize_filename(feat)}_outside_Deviations_boxplot.png"
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, fn))
                plt.close()

def plot_cross_zone_comparison(zone_dfs, feature_template, zones, deviation_type,
                               tolerance, output_dir):
    """
    Generate box plot comparing a feature across multiple zones during specified deviation.
    feature_template: string with '{zone}' placeholder for zone-specific column names.
    deviation_type: 'up' or 'down'.
    """
    records = []
    feat_name = feature_template.replace('{zone}', '').strip()
    # Determine title and filename suffix
    if deviation_type == 'up':
        title_suffix = 'Up-Regulation Deviations'
        file_suffix = 'UpDev'
    elif deviation_type == 'down':
        title_suffix = 'Down-Regulation Deviations'
        file_suffix = 'DownDev'
    elif deviation_type == 'up_match':
        title_suffix = 'Up-Matches (No Deviation)'
        file_suffix = 'UpMatch'
    elif deviation_type == 'down_match':
        title_suffix = 'Down-Matches (No Deviation)'
        file_suffix = 'DownMatch'
    else:
        print(f"[Error] Unknown deviation_type '{deviation_type}', skipping.")
        return
    # Collect data per zone
    for zone in zones:
        df = zone_dfs.get(zone)
        if df is None:
            print(f"[Warning] No data for zone {zone}, skipping.")
            continue
        feat_col = feature_template.format(zone=zone)
        if feat_col not in df.columns:
            print(f"[Warning] Feature '{feat_col}' not found in zone {zone}, skipping.")
            continue
        spot_col = f"pri {zone} spot €/mwh cet h a"
        up_col = f"pri {zone} regulation up €/mwh cet min15 a"
        down_col = f"pri {zone} regulation down €/mwh cet min15 a"
        # Compute mask per deviation_type
        if deviation_type == 'up':
            mask = ~np.isclose(df[up_col], df[spot_col], atol=tolerance)
        elif deviation_type == 'down':
            mask = ~np.isclose(df[down_col], df[spot_col], atol=tolerance)
        elif deviation_type == 'up_match':
            mask = np.isclose(df[up_col], df[spot_col], atol=tolerance)
        else:  # 'down_match'
            mask = np.isclose(df[down_col], df[spot_col], atol=tolerance)
        sub = df.loc[mask, [feat_col]].dropna()
        if sub.empty:
            continue
        sub = sub.rename(columns={feat_col: 'feature_value'})
        sub['Zone'] = zone.upper()
        records.append(sub)
    if not records:
        print(f"[Warning] No data for cross-zone plotting of {feat_name}, skipping.")
        return
    combined = pd.concat(records, ignore_index=True)
    # Plot boxplot
    plt.figure(figsize=(8, 6))
    order = [z.upper() for z in zones]
    sns.boxplot(data=combined, x='Zone', y='feature_value', order=order)
    plt.title(f"Comparison of {feat_name} during {title_suffix} across Zones")
    plt.xlabel('Price Zone')
    plt.ylabel(feat_name)
    ensure_dir(output_dir)
    fname = f"{sanitize_filename(feat_name)}_{file_suffix}_ZoneCompare_boxplot.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def plot_target_distribution(zone_dfs, output_dir):
    """
    Plot boxplots (with jitter) of up- and down-regulation prices across zones.
    Saves two figures: Up_Price_by_Zone_boxplot.png and Down_Price_by_Zone_boxplot.png
    """
    ensure_dir(output_dir)
    # Up price distribution
    up_records = []
    for zone, df in zone_dfs.items():
        col = f"pri {zone} regulation up €/mwh cet min15 a"
        if col not in df.columns:
            continue
        tmp = df[[col]].dropna().copy()
        tmp.rename(columns={col: 'price'}, inplace=True)
        tmp['Zone'] = zone.upper()
        up_records.append(tmp)
    if up_records:
        up_df = pd.concat(up_records, ignore_index=True)
        # Winsorize to 5th–95th percentile
        p5, p95 = up_df['price'].quantile([0.05, 0.95])
        up_df['price_capped'] = up_df['price'].clip(lower=p5, upper=p95)
        plt.figure(figsize=(8, 6))
        # Violin plot scaled by count, inner quartiles
        sns.violinplot(
            data=up_df, x='Zone', y='price_capped',
            inner='quartile', scale='count', palette='Set3'
        )
        # Overlay low-opacity strip plot
        plt.title('Up-Regulation Price Distribution by Zone')
        plt.xlabel('Price Zone')
        plt.ylabel('Up Price (capped 5th–95th pct)')
        fup = os.path.join(output_dir, 'Up_Price_by_Zone_boxplot.png')
        plt.tight_layout()
        plt.savefig(fup)
        plt.close()
    # Down price distribution
    down_records = []
    for zone, df in zone_dfs.items():
        col = f"pri {zone} regulation down €/mwh cet min15 a"
        if col not in df.columns:
            continue
        tmp = df[[col]].dropna().copy()
        tmp.rename(columns={col: 'price'}, inplace=True)
        tmp['Zone'] = zone.upper()
        down_records.append(tmp)
    if down_records:
        down_df = pd.concat(down_records, ignore_index=True)
        # Winsorize to 5th–95th percentile
        p5d, p95d = down_df['price'].quantile([0.05, 0.95])
        down_df['price_capped'] = down_df['price'].clip(lower=p5d, upper=p95d)
        plt.figure(figsize=(8, 6))
        sns.violinplot(
            data=down_df, x='Zone', y='price_capped',
            inner='quartile', scale='count', palette='Set3'
        )
        plt.title('Down-Regulation Price Distribution by Zone')
        plt.xlabel('Price Zone')
        plt.ylabel('Down Price (capped 5th–95th pct)')
        fdn = os.path.join(output_dir, 'Down_Price_by_Zone_boxplot.png')
        plt.tight_layout()
        plt.savefig(fdn)
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Generate additional EDA comparison plots for deviation analysis."
    )
    parser.add_argument(
        "--data-folder", default="/Volumes/T9",
        help="Base folder containing area subfolders with test_rounds."
    )
    parser.add_argument(
        "--output-folder", default="chapters/plots/results/eda_comparison",
        help="Directory to save comparison plots."
    )
    parser.add_argument(
        "--tolerance", type=float, default=0,
        help="Tolerance for price matching (default: 0)."
    )
    parser.add_argument(
        "--zones", nargs='+', default=['no1', 'no2', 'no3', 'no4', 'no5'],
        help="List of zones to include (default: all)."
    )
    args = parser.parse_args()
    data_folder = args.data_folder
    output_folder = args.output_folder
    tol = args.tolerance
    zones = args.zones
    # Prepare output directory
    ensure_dir(output_folder)
    # Load EDA data for each zone
    zone_dfs = {}
    for zone in zones:
        # Look for test data rounds
        test_dir = os.path.join(data_folder, zone, 'test_rounds')
        x_file, y_file, rnd = find_last_round_files(test_dir)
        if not x_file or not y_file:
            print(f"[Warning] Missing data for zone {zone}, skipping.")
            continue
        df = load_and_prepare_data(x_file, y_file)
        if df is None or df.empty:
            print(f"[Warning] Empty or invalid DataFrame for zone {zone}, skipping.")
            continue
        zone_dfs[zone] = df
    # Instruction Set 1: Combined Deviation State Plots (Within Zone)
    if 'no1' in zone_dfs:
        plot_within_zone_deviation(zone_dfs['no1'], 'no1', tol, output_folder)
    else:
        print("[Info] Zone 'no1' data not available for within-zone plots.")
    # Instruction Set 2: Cross-Zone Heterogeneity Plots
    # Case A: Spot Price during Up-Regulation Deviations across NO1, NO4, NO5
    # Case A1: Spot Price during Up-Regulation Deviations across zones
    plot_cross_zone_comparison(
        zone_dfs,
        feature_template="pri {zone} spot €/mwh cet h a",
        zones=['no1', 'no2', 'no3', 'no4', 'no5'],
        deviation_type='up',
        tolerance=tol,
        output_dir=output_folder
    )
    # Case A2: Spot Price during Down-Regulation Deviations across zones
    plot_cross_zone_comparison(
        zone_dfs,
        feature_template="pri {zone} spot €/mwh cet h a",
        zones=['no1', 'no2', 'no3', 'no4', 'no5'],
        deviation_type='down',
        tolerance=tol,
        output_dir=output_folder
    )
    # Case A3: Spot Price during Up-Matches (no deviation) across zones
    plot_cross_zone_comparison(
        zone_dfs,
        feature_template="pri {zone} spot €/mwh cet h a",
        zones=['no1', 'no2', 'no3', 'no4', 'no5'],
        deviation_type='up_match',
        tolerance=tol,
        output_dir=output_folder
    )
    # Case A4: Spot Price during Down-Matches (no deviation) across zones
    plot_cross_zone_comparison(
        zone_dfs,
        feature_template="pri {zone} spot €/mwh cet h a",
        zones=['no1', 'no2', 'no3', 'no4', 'no5'],
        deviation_type='down_match',
        tolerance=tol,
        output_dir=output_folder
    )
    # Case B: Wind Production during Up-Regulation Deviations across NO1- NO4
    plot_cross_zone_comparison(
        zone_dfs,
        feature_template="pro {zone} wnd mwh/h cet min15 a",
        zones=['no1', 'no2', 'no3', 'no4'],
        deviation_type='up',
        tolerance=tol,
        output_dir=output_folder
    )
    # Case C: RDL during Up-Regulation Deviations across zones
    plot_cross_zone_comparison(
        zone_dfs,
        feature_template="rdl {zone} mwh/h cet min15 a",
        zones=['no1', 'no2', 'no3', 'no4', 'no5'],
        deviation_type='up',
        tolerance=tol,
        output_dir=output_folder
    )
    # Plot target distributions across zones (up & down prices)
    plot_target_distribution(zone_dfs, output_folder)
    # Plot match vs deviation distributions per feature in a 2x2 grid (up/down × match/dev)
    features = [
        ('spot',     'pri {zone} spot €/mwh cet h a'),
        ('rdl',      'rdl {zone} mwh/h cet min15 a'),
        ('intraday','con {zone} intraday mwh/h cet h a'),
        ('hydro',    'pro {zone} hydro tot mwh/h cet h af'),
        ('wind',     'pro {zone} wnd mwh/h cet min15 a'),
        ('cooling',  'con {zone} cooling % cet min15 s'),
        ('heating',  'con {zone} heating % cet min15 s'),
    ]
    for key, templ in features:
        # Prepare DataFrames for each quadrant
        mats = {}
        for direction in ['up', 'down']:
            # match
            recs = []
            for zone, df in zone_dfs.items():
                feat_col = templ.format(zone=zone)
                if feat_col not in df.columns:
                    continue
                spot_col = f"pri {zone} spot €/mwh cet h a"
                tgt_col = f"pri {zone} regulation {direction} €/mwh cet min15 a"
                mask = np.isclose(df[tgt_col], df[spot_col], atol=tol)
                tmp = df.loc[mask, [feat_col]].dropna().copy()
                tmp.columns = ['Value']
                tmp['Zone'] = zone.upper()
                recs.append(tmp)
            mats[f"{direction}_match"] = pd.concat(recs, ignore_index=True) if recs else None
            # deviation
            recs = []
            for zone, df in zone_dfs.items():
                feat_col = templ.format(zone=zone)
                if feat_col not in df.columns:
                    continue
                spot_col = f"pri {zone} spot €/mwh cet h a"
                tgt_col = f"pri {zone} regulation {direction} €/mwh cet min15 a"
                mask = ~np.isclose(df[tgt_col], df[spot_col], atol=tol)
                tmp = df.loc[mask, [feat_col]].dropna().copy()
                tmp.columns = ['Value']
                tmp['Zone'] = zone.upper()
                recs.append(tmp)
            mats[f"{direction}_dev"] = pd.concat(recs, ignore_index=True) if recs else None
        # Build 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
        labels = [('up_match', f"Up Match: {key}"),
                  ('up_dev',   f"Up Deviation: {key}"),
                  ('down_match', f"Down Match: {key}"),
                  ('down_dev',   f"Down Deviation: {key}")]
        for idx, (k, title) in enumerate(labels):
            ax = axes[idx//2][idx%2]
            dfq = mats.get(k)
            if dfq is not None and not dfq.empty:
                # Violin plot for distribution
                sns.violinplot(
                    data=dfq, x='Zone', y='Value',
                    inner='quartile', scale='count', palette='Set3', ax=ax
                )
                ax.set_title(title)
                ax.set_xlabel('')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(title)
                ax.set_xticks([])
            # Y-label only on left column
            if idx % 2 == 0:
                ax.set_ylabel(key)
            else:
                ax.set_ylabel('')
        fig.suptitle(f"{key.upper()} Distributions: Match vs Deviation by Zone")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"{key}_match_dev_2x2_boxplot.png"
        plt.savefig(os.path.join(output_folder, fname))
        plt.close(fig)

if __name__ == '__main__':
    main()