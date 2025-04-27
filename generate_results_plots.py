# Script to generate plots and tables for Chapter 4 (Results) of the thesis.
import os
import json
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

def make_table(df: pd.DataFrame, direction: str, model_order, model_names, areas) -> str:
    """Generate LaTeX code for one table (direction='up' or 'down')."""
    sub = df[df['direction'] == direction]
    lines = []
    lines.append(r"\\begin{table}")
    lines.append(r"\\centering")
    lines.append(f"\\caption{{Detailed Metrics {direction.capitalize()} Test}}")
    lines.append(f"\\label{{tab:metrics_{direction}_test}}")
    # Include extra model-agnostic columns and adjust column spec
    extra_cols = ["n_original", "n_filtered", "pct_removed"]
    # Column spec: area (l), extra columns (r x len(extra_cols)), then metrics for each model
    col_spec = "|l|" + "r" * len(extra_cols) + "|" + "".join(["rrr|" for _ in model_order])
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\\toprule")
    # Header row: area and extra columns, then model groupings
    header = "Area"
    for col in extra_cols:
        header += f" & {col}"
    for m in model_order:
        header += f" & \\multicolumn{{3}}{{c|}}{{{model_names[m]}}}"
    header += r" \\\\"
    lines.append(header)
    # Metric names row: blank for extra columns, then metrics for each model
    metric_row = "Metric"
    for _ in extra_cols:
        metric_row += " &"
    for _ in model_order:
        metric_row += " & MAE & R2 & RMSE"
    metric_row += r" \\\\"
    lines.append(r"\\midrule")
    lines.append(metric_row)
    # Data rows
    for area in areas:
        # get extra column values (same for all models for this area/direction)
        extra = sub[sub['area'] == area].iloc[0]
        row = [
            area,
            str(int(extra['n_original'])),
            str(int(extra['n_filtered'])),
            f"{extra['pct_removed']:.2f}"
        ]
        for m in model_order:
            r = sub[(sub['area'] == area) & (sub['model'] == m)].iloc[0]
            row += [f"{r['mae']:.4f}", f"{r['r2']:.4f}", f"{r['rmse']:.4f}"]
        lines.append("   " + " & ".join(row) + r" \\\\")
    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(r"\\end{table}")
    return "\n".join(lines)

DATASET = None
from interpret.glassbox import ExplainableBoostingRegressor

def load_metrics(metrics_path):
    with open(metrics_path, 'r') as f:
        return json.load(f)

def create_output_dirs(base_dir):
    dirs = {
        'performance': os.path.join(base_dir, 'performance'),
        'detailed_tables': os.path.join(base_dir, 'performance', 'tables'),
        'ebm': os.path.join(base_dir, 'ebm'),
        'xgb': os.path.join(base_dir, 'xgb'),
        'stacked': os.path.join(base_dir, 'stacked'),
        'timeseries': os.path.join(base_dir, 'timeseries'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def plot_overall_performance(metrics, zones, targets, dirs):
    # Bar charts of avg R2, MAE, RMSE per model (avg across zones)
    for target in targets:
        avg_metrics = {}
        for model, data in metrics.items():
            vals = [data[z][target] for z in zones]
            mae = np.mean([v['MAE'] for v in vals])
            rmse = np.mean([v['RMSE'] for v in vals])
            r2 = np.mean([v['R2'] for v in vals])
            avg_metrics[model] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        df = pd.DataFrame(avg_metrics).T
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, metric in zip(axes, ['R2', 'MAE', 'RMSE']):
            sns.barplot(x=df.index, y=df[metric], ax=ax)
            ax.set_title(f"{metric} for {target.capitalize()} (avg over zones, {DATASET})")
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['performance'], f'overall_performance_{target}_{DATASET}.png'))
        plt.close()

def save_detailed_tables(metrics, zones, targets, dirs):
    # Save CSV and Markdown tables of detailed metrics per zone and model
    for target in targets:
        cols = pd.MultiIndex.from_product([metrics.keys(), ['MAE', 'RMSE', 'R2']], names=['Model','Metric'])
        df = pd.DataFrame(index=zones, columns=cols)
        for model in metrics.keys():
            for zone in zones:
                vals = metrics[model][zone][target]
                df.loc[zone, (model, 'MAE')] = vals['MAE']
                df.loc[zone, (model, 'RMSE')] = vals['RMSE']
                df.loc[zone, (model, 'R2')] = vals['R2']
        df = df.sort_index(axis=1)
        csv_path = os.path.join(dirs['detailed_tables'], f'detailed_metrics_{target}_{DATASET}.csv')
        df.to_csv(csv_path)
        with open(os.path.join(dirs['detailed_tables'], f'detailed_metrics_{target}_{DATASET}.md'), 'w') as f:
            f.write(df.to_markdown())

def plot_performance_gain(metrics, zones, targets, dirs):
    # % improvement in MAE/RMSE over baseline (e.g., Naive)
    # Identify baseline model key
    baseline_candidates = [m for m in metrics if 'naive' in m.lower()]
    if not baseline_candidates:
        raise ValueError("No baseline 'Naive' model found in metrics")
    baseline_key = baseline_candidates[0]
    base = metrics[baseline_key]
    for target in targets:
        gain = {}
        for model in metrics:
            if model == baseline_key:
                continue
            gain[model] = {}
            for metric in ['MAE', 'RMSE']:
                base_vals = [base[z][target][metric] for z in zones]
                mod_vals = [metrics[model][z][target][metric] for z in zones]
                improvements = []
                for b, m in zip(base_vals, mod_vals):
                    if b and b != 0:
                        improvements.append((b - m) / b * 100)
                    else:
                        improvements.append(0.0)
                pct = np.mean(improvements) if improvements else 0.0
                gain[model][metric] = pct
        df = pd.DataFrame(gain).T
        fig, ax = plt.subplots(figsize=(8, 6))
        df.plot(kind='bar', ax=ax)
        ax.set_title(f"% Improvement over {baseline_key} for {target.capitalize()} ({DATASET})")
        ax.set_ylabel('% Improvement')
        ax.set_xlabel('Model')
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['performance'], f'performance_gain_{target}_{DATASET}.png'))
        plt.close()

def plot_ebm_importance(ebm, dirs, zone, target, top_n=10):
    # Global feature importance from EBM
    exp = ebm.explain_global()
    data = exp.data()
    df = pd.DataFrame({'feature': data['names'], 'importance': data['scores']})
    df = df.sort_values('importance', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df, palette='viridis')
    plt.title(f'EBM Global Feature Importance ({zone} {target}, {DATASET})')
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['ebm'], f'ebm_global_importance_{zone}_{target}_{DATASET}.png'))
    plt.close()
    return data

def plot_ebm_shape_functions(ebm, exp_data, dirs, zone, target, top_n=7, bottom_n=0):
    # Univariate shape functions for top terms
    # Retrieve bin edges and term information; handle attribute naming differences
    try:
        bin_edges = ebm.feature_bins_
    except AttributeError:
        try:
            bin_edges = ebm.bins_
        except AttributeError as e:
            raise RuntimeError("EBM model missing bin edges attribute (tried 'feature_bins_' and 'bins_')") from e
    try:
        term_scores = ebm.term_scores_
        term_features = ebm.term_features_
    except AttributeError as e:
        raise RuntimeError("EBM model missing term_scores_ or term_features_ attributes") from e
    names = exp_data['names']
    scores = exp_data['scores']
    # Identify all univariate terms (single-feature)
    all_uni = [(i, s) for i, s in enumerate(scores) if len(term_features[i]) == 1]
    # Sort by importance descending
    uni_sorted = sorted(all_uni, key=lambda x: x[1], reverse=True)
    # Select top_n most important univariates
    if top_n is None or top_n >= len(uni_sorted):
        top_uni = uni_sorted
    else:
        top_uni = uni_sorted[:top_n]
    # Select bottom_n least important univariates
    bottom_uni = []
    if bottom_n:
        if bottom_n >= len(uni_sorted):
            bottom_uni = uni_sorted
        else:
            bottom_uni = uni_sorted[-bottom_n:]
    # Combine selections (top first, then bottom, avoiding duplicates)
    selected = top_uni.copy()
    top_indices = {i for i, _ in top_uni}
    for i_s in bottom_uni:
        if i_s[0] not in top_indices:
            selected.append(i_s)
    # Plot each selected univariate shape
    for i, _ in selected:
        fidx = term_features[i][0]
        # Ensure bin edges are a flat numpy array
        edges_arr = np.asarray(bin_edges[fidx])
        if edges_arr.ndim > 1:
            edges_arr = edges_arr.flatten()
        # Need at least two edges to compute mids
        if edges_arr.size < 2:
            print(f"Warning: Not enough bin edges for feature '{names[fidx]}', skipping shape plot.")
            continue
        mids = (edges_arr[:-1] + edges_arr[1:]) / 2
        ys_arr = np.asarray(term_scores[i])
        # Align lengths
        L = min(len(mids), len(ys_arr))
        if L == 0:
            print(f"Warning: Empty values for feature '{names[fidx]}', skipping shape plot.")
            continue
        mids = mids[:L]
        ys_arr = ys_arr[:L]
        plt.figure(figsize=(8, 4))
        plt.plot(mids, ys_arr, marker='o')
        plt.title(f'Shape: {names[i]} ({zone} {target}, {DATASET})')
        plt.xlabel(names[i])
        plt.ylabel('f(x)')
        plt.tight_layout()
        # Sanitize feature name for filename
        fn = names[i].replace(' ', '_')
        # Replace any non-alphanumeric, non _ . - characters with underscore
        fn = re.sub(r'[^A-Za-z0-9_.-]', '_', fn)
        plt.savefig(os.path.join(dirs['ebm'], f'ebm_shape_{fn}_{zone}_{target}_{DATASET}.png'))
        plt.close()

def plot_ebm_interactions(ebm, exp_data, dirs, zone, target, top_n=2):
    # Pairwise interaction heatmaps
    # Retrieve bin edges and term information; handle attribute naming differences
    try:
        bin_edges = ebm.feature_bins_
    except AttributeError:
        try:
            bin_edges = ebm.bins_
        except AttributeError:
            return
    try:
        term_scores = ebm.term_scores_
        term_features = ebm.term_features_
    except AttributeError:
        return
    names = exp_data['names']
    scores = exp_data['scores']
    inter = [(i, s) for i, s in enumerate(scores) if len(term_features[i]) == 2]
    inter = sorted(inter, key=lambda x: x[1], reverse=True)[:top_n]
    for i, _ in inter:
        f1, f2 = term_features[i]
        # Ensure bin edges are flat numpy arrays
        e1 = np.asarray(bin_edges[f1])
        e2 = np.asarray(bin_edges[f2])
        if e1.ndim > 1:
            e1 = e1.flatten()
        if e2.ndim > 1:
            e2 = e2.flatten()
        # Need at least two edges for each
        if e1.size < 2 or e2.size < 2:
            print(f"Warning: Not enough bin edges for interaction term '{names[i]}', skipping.")
            continue
        m1 = (e1[:-1] + e1[1:]) / 2
        m2 = (e2[:-1] + e2[1:]) / 2
        Z_arr = np.asarray(term_scores[i])
        # Check shape alignment
        if Z_arr.ndim != 2 or Z_arr.shape != (len(m1), len(m2)):
            print(f"Warning: Interaction shape mismatch for term '{names[i]}': Z shape {Z_arr.shape}, expected ({len(m1)},{len(m2)}). Skipping.")
            continue
        plt.figure(figsize=(6, 5))
        sns.heatmap(Z_arr, xticklabels=np.round(m2, 2), yticklabels=np.round(m1, 2), cmap='coolwarm')
        plt.title(f'Interaction: {names[i]} ({zone} {target}, {DATASET})')
        plt.xlabel(f'Feat {f2}')
        plt.ylabel(f'Feat {f1}')
        plt.tight_layout()
        # Sanitize interaction name for filename
        fn = names[i].replace(' ', '_')
        # Replace any non-alphanumeric, non _ . - characters with underscore
        fn = re.sub(r'[^A-Za-z0-9_.-]', '_', fn)
        plt.savefig(os.path.join(dirs['ebm'], f'ebm_inter_{fn}_{zone}_{target}_{DATASET}.png'))
        plt.close()

def plot_xgb_importance(xgb_model, dirs, zone, target, top_n=10):
    # XGBoost feature importance (gain)
    try:
        booster = xgb_model.get_booster()
        imp = booster.get_score(importance_type='gain')
    except Exception:
        imp = dict(zip(xgb_model.feature_names_in_, xgb_model.feature_importances_))
    imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
    names = list(imp.keys())[:top_n]
    scores = [imp[n] for n in names]
    df = pd.DataFrame({'feature': names, 'importance': scores})
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df, palette='magma')
    plt.title(f'XGBoost Importance ({zone} {target}, {DATASET})')
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['xgb'], f'xgb_import_{zone}_{target}_{DATASET}.png'))
    plt.close()

def stacked_meta_performance(df, dirs, zone, target):
    # Metrics for meta-learner predicting EBM residuals
    actual = df['actual']
    ebm_pred = df['ebm_pred']
    stacked_pred = df['predicted']
    res_act = actual - ebm_pred
    res_pred = stacked_pred - ebm_pred
    mae = mean_absolute_error(res_act, res_pred)
    rmse = np.sqrt(mean_squared_error(res_act, res_pred))
    r2 = r2_score(res_act, res_pred)
    out = f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR2: {r2:.3f}\n"
    with open(os.path.join(dirs['stacked'], f'meta_perf_{zone}_{target}_{DATASET}.txt'), 'w') as f:
        f.write(out)
    print(f"Meta-learner performance ({zone} {target}, {DATASET}): {out}")
    return res_act, res_pred

def plot_stacked_meta_importance(meta_model, dirs, zone, target, top_n=10):
    # Feature importance of meta-learner
    try:
        booster = meta_model.get_booster()
        imp = booster.get_score(importance_type='gain')
    except Exception:
        imp = dict(zip(meta_model.feature_names_in_, meta_model.feature_importances_))
    imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
    names = list(imp.keys())[:top_n]
    scores = [imp[n] for n in names]
    df = pd.DataFrame({'feature': names, 'importance': scores})
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df, palette='plasma')
    plt.title(f'Meta-learner Importance ({zone} {target}, {DATASET})')
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['stacked'], f'meta_import_{zone}_{target}_{DATASET}.png'))
    plt.close()

def plot_time_series(df, dirs, zone, target, start_date=None, days=7):
    # Actual vs predicted over a short period
    # Ensure timestamps are parsed as datetimes (may be tz-aware)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Capture timezone info if present for later alignment
    tz = df['timestamp'].dt.tz
    # Determine start datetime for the series
    if start_date is None:
        start_dt = df['timestamp'].min()
    else:
        start_dt = pd.to_datetime(start_date)
    # If timestamps are tz-aware and start_dt is naive, localize to match
    if tz is not None and start_dt.tzinfo is None:
        start_dt = start_dt.tz_localize(tz)
    # Compute end datetime relative to start
    end_dt = start_dt + pd.Timedelta(days=days)
    mask = (df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)
    df_p = df.loc[mask]
    # Plot with styled lines and date formatting
    fig, ax = plt.subplots(figsize=(12, 6))
    styles = {
        'actual':  {'ls': '-',  'lw': 2},
        'naive':   {'ls': '--', 'lw': 1},
        'xgb':     {'ls': ':',  'lw': 1},
        'ebm':     {'ls': '-',  'lw': 1},
        'stacked': {'ls': '--', 'lw': 1}
    }
    plot_order = [
        ('actual', 'actual', 'Actual'),
        ('naive_pred', 'naive', 'Naive'),
        ('standalone_xgb_pred', 'xgb', 'XGBoost'),
        ('ebm_pred', 'ebm', 'EBM'),
        ('predicted', 'stacked', 'Stacked')
    ]
    for col_key, style_key, label in plot_order:
        if col_key in df_p.columns:
            ax.plot(
                df_p['timestamp'], df_p[col_key], label=label,
                linestyle=styles[style_key]['ls'],
                linewidth=styles[style_key]['lw']
            )
    n_points = len(df_p)
    ax.set_title(f'Actual vs Predictions ({zone} {target}, {DATASET}; {n_points} points)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    # Date formatting & rotation
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.legend()
    # Compute metrics for the plotted period
    metrics_lines = []
    model_map = {
        'Stacked': 'predicted',
        'EBM': 'ebm_pred',
        'XGBoost': 'standalone_xgb_pred',
        'Naive': 'naive_pred'
    }
    for name, col in model_map.items():
        if col in df_p.columns:
            y_true = df_p['actual']
            y_pred = df_p[col]
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            metrics_lines.append(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
    if metrics_lines:
        textstr = "\n".join(metrics_lines)
        ax.text(
            0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5)
        )
    plt.tight_layout()
    # Include start date in filename to avoid overwriting
    date_str = start_dt.strftime('%Y%m%d')
    filename = f'time_series_{zone}_{target}_{date_str}_{DATASET}.png'
    fig.savefig(os.path.join(dirs['timeseries'], filename))
    plt.close(fig)
    return df_p

def plot_ebm_residuals(df_p, dirs, zone, target):
    # Residuals of EBM over the period
    resid = df_p['actual'] - df_p['ebm_pred']
    plt.figure(figsize=(12, 4))
    plt.plot(df_p['timestamp'], resid)
    plt.title(f'EBM Residuals ({zone} {target}, {DATASET})')
    plt.xlabel('Time')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['timeseries'], f'residuals_{zone}_{target}_{DATASET}.png'))
    plt.close()

def main():
    areas = ['no1', 'no2', 'no3', 'no4', 'no5']
    targets = ['up', 'down']
    for area in areas:
        for target in targets:
            # Generate plots for each area and target
            loop(area, target)

def loop(representative_zone, representative_target):
    # Configurable representative case
    representative_zone = representative_zone.lower()
    representative_target = representative_target.lower()
    TEST = "test"
    VALIDATION = "validation"
    val_or_test = TEST
    # Set global dataset identifier for titles and filenames
    global DATASET
    DATASET = val_or_test
    # Paths
    metrics_path = f'./results/{val_or_test}_set_metrics.json'
    preds_path = f'./results/predictions/{val_or_test}/{representative_zone}_{representative_target}_predictions.csv'
    # Paths to the final-fold models (replace .joblib with .pkl if needed)
    ebm_model_path = f'./models/{val_or_test}/ebm_last_run_{representative_zone}_{representative_target}.joblib'
    xgb_model_path = f'./models/{val_or_test}/xgb_last_run_{representative_zone}_{representative_target}.joblib'
    stacked_meta_path = f'./models/{val_or_test}/stacked_last_run_{representative_zone}_{representative_target}.joblib'
    # Load metrics and determine models, zones, targets from JSON structure
    raw_metrics = load_metrics(metrics_path)
    # raw_metrics[zone][target][model] = metrics
    zones = list(raw_metrics.keys())
    if not zones:
        raise ValueError(f'No zones found in metrics file: {metrics_path}')
    targets = list(raw_metrics[zones[0]].keys())
    if not targets:
        raise ValueError(f'No targets found under zone {zones[0]} in metrics file')
    models = list(raw_metrics[zones[0]][targets[0]].keys())
    if not models:
        raise ValueError(f'No models found under zone {zones[0]}, target {targets[0]} in metrics file')
    # Pivot to metrics[model][zone][target]
    metrics = {}
    for model in models:
        metrics[model] = {}
        for zone in zones:
            metrics[model][zone] = {}
            for target in targets:
                try:
                    metrics[model][zone][target] = raw_metrics[zone][target][model]
                except KeyError:
                    raise KeyError(f"Model '{model}' missing for zone '{zone}', target '{target}' in metrics file")
    # Prepare output dirs under a subfolder for the representative zone/target
    base_root = f'./chapters/{val_or_test}/plots'
    rep_folder = f"{representative_zone}_{representative_target}"
    base = os.path.join(base_root, rep_folder)
    dirs = create_output_dirs(base)
    # Generate performance plots and tables
    plot_overall_performance(metrics, zones, targets, dirs)
    save_detailed_tables(metrics, zones, targets, dirs)
    plot_performance_gain(metrics, zones, targets, dirs)
    # Load representative models
    ebm = joblib.load(ebm_model_path)
    xgb = joblib.load(xgb_model_path)
    stacked_meta = joblib.load(stacked_meta_path)
    # EBM explainability
    exp_data = plot_ebm_importance(ebm, dirs, representative_zone, representative_target)
    # Plot all univariate shape functions for EBM
    plot_ebm_shape_functions(
        ebm, exp_data, dirs,
        representative_zone, representative_target,
        top_n=None
    )
    plot_ebm_interactions(ebm, exp_data, dirs, representative_zone, representative_target)
    # XGBoost importance
    plot_xgb_importance(xgb, dirs, representative_zone, representative_target)
    # Stacked meta-learner performance & importance
    df_pred = pd.read_csv(preds_path)
    # Load additional model predictions (XGBoost and Naive) for time series plots
    # Derive file paths based on the stacked predictions path
    xgb_preds_path = preds_path.replace('_predictions.csv', '_xgb_predictions.csv')
    if os.path.exists(xgb_preds_path):
        df_xgb = pd.read_csv(xgb_preds_path)
        df_pred = df_pred.merge(
            df_xgb[['timestamp', 'predicted']], on='timestamp', how='left', suffixes=('', '_xgb')
        )
        df_pred.rename(columns={'predicted_xgb': 'standalone_xgb_pred'}, inplace=True)
    else:
        print(f"Warning: XGBoost predictions file not found: {xgb_preds_path}")
    naive_preds_path = preds_path.replace('_predictions.csv', '_naive_predictions.csv')
    if os.path.exists(naive_preds_path):
        df_naive = pd.read_csv(naive_preds_path)
        df_pred = df_pred.merge(
            df_naive[['timestamp', 'predicted']], on='timestamp', how='left', suffixes=('', '_naive')
        )
        df_pred.rename(columns={'predicted_naive': 'naive_pred'}, inplace=True)
    else:
        print(f"Warning: Naive predictions file not found: {naive_preds_path}")
    stacked_meta_performance(df_pred, dirs, representative_zone, representative_target)
    plot_stacked_meta_importance(stacked_meta, dirs, representative_zone, representative_target)
    # Time-series example and residuals
    if val_or_test == TEST:
        df_period = plot_time_series(df_pred, dirs, representative_zone, representative_target, start_date="2024-07-10", days=15)
        df_period = plot_time_series(df_pred, dirs, representative_zone, representative_target, start_date="2024-10-01", days=15)
        df_period = plot_time_series(df_pred, dirs, representative_zone, representative_target, start_date="2025-01-01", days=15)
        df_period = plot_time_series(df_pred, dirs, representative_zone, representative_target, start_date="2025-03-01", days=15)


        #plot_ebm_residuals(df_period, dirs, representative_zone, representative_target)
    print('All plots and tables saved to', base)
    try:
        # Path to filtered metrics summary
        filtered_csv = f'./results/predictions/{val_or_test}_filtered/filtered_metrics_summary.csv'
        df_filt = pd.read_csv(filtered_csv)
        # Configuration
        model_order = ["naive", "xgb", "ebm", "stacked"]
        model_names = {"naive": "naive", "xgb": "xgboost", "ebm": "ebm", "stacked": "stacked"}
        # Areas sorted naturally (no1, no2, ...)
        areas = sorted(df_filt['area'].unique(), key=lambda s: int(s.replace('no', '')))
        # Build tables
        output = ["\\begin{landscape}", ""]
        output.append(make_table(df_filt, "up", model_order, model_names, areas))
        output.append("")
        output.append(make_table(df_filt, "down", model_order, model_names, areas))
        output.append("")
        output.append("\\end{landscape}")
        latex_str = "\n".join(output)
        # Save LaTeX to file
        tex_out = os.path.join(base, 'filtered_metrics_tables.tex')
        with open(tex_out, 'w') as f:
            f.write(latex_str)
        print(f'Wrote LaTeX filtered metrics tables to {tex_out}')
    except Exception as e:
        print(f'Error generating LaTeX tables: {e}', file=sys.stderr)

    # Plot filtered predictions time series where spot price != target (limited window & styled lines)
    try:
        # Load filtered prediction CSVs for each model
        filtered_dir = f'./results/predictions/{val_or_test}_filtered'
        models_filtered = ['naive', 'xgb', 'ebm', 'stacked']
        dfs = {}
        for m in models_filtered:
            fpath = os.path.join(filtered_dir, f"{representative_zone}_{representative_target}_{m}_filtered.csv")
            df_m = pd.read_csv(fpath, parse_dates=['timestamp'])
            df_m = df_m[['timestamp', 'actual', 'predicted']].rename(columns={'predicted': m})
            dfs[m] = df_m
        # Merge on timestamp (naive carries actual)
        df_plot = dfs['naive'][['timestamp', 'actual', 'naive']].copy()
        for m in ['xgb', 'ebm', 'stacked']:
            df_plot = df_plot.merge(dfs[m][['timestamp', m]], on='timestamp', how='inner')
        df_plot = df_plot.sort_values('timestamp')
        N_days = 30
        start_ts = df_plot['timestamp'].min()
        df_plot = df_plot[df_plot['timestamp'] <= start_ts + pd.Timedelta(days=N_days)]
        # Facet by model (small multiples), no interpolation
        # Facet by model (small multiples), no interpolation, share y-axis, include actuals, gridlines
        models = ['Naive', 'XGBoost', 'EBM', 'Stacked']
        pred_cols = ['naive', 'xgb', 'ebm', 'stacked']
        n_points = len(df_plot)
        fig, axes = plt.subplots(
            len(models), 1,
            sharex=True, sharey=True,
            figsize=(12, 9), constrained_layout=True
        )
        # Plot each model in its own row, with shared y-axis and gridlines
        for ax, name, col in zip(axes, models, pred_cols):
            # model predictions (orange, colorblind-friendly)
            ax.scatter(
                df_plot['timestamp'], df_plot[col],
                s=6, alpha=0.6, color='#D55E00'
            )
            # faint reference to actual values (smaller and lighter)
            ax.scatter(
                df_plot['timestamp'], df_plot['actual'],
                s=2, alpha=0.15, color='gray'
            )
            # compute error metrics for this model
            y_true = df_plot['actual']
            y_pred = df_plot[col]
            mae = mean_absolute_error(y_true, y_pred)
            rmse = (mean_squared_error(y_true, y_pred) ** 0.5)
            r2 = r2_score(y_true, y_pred)
            # annotate metrics
            ax.text(
                0.01, 0.95,
                f'MAE={mae:.1f}\nRMSE={rmse:.1f}\nR²={r2:.2f}',
                transform=ax.transAxes,
                va='top', ha='left', fontsize='small',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
            )
            ax.set_ylabel(name)
            # gridlines
            ax.grid(axis='y', color='#ddd', linestyle='--', linewidth=0.5)
            ax.grid(axis='x', color='#eee', linestyle='--', linewidth=0.3)
        # global labels
        axes[-1].set_xlabel('Time')
        # refine x-axis ticks: one per week
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.setp(axes[-1].get_xticklabels(), rotation=30, ha='right')
        axes[-1].set_xlabel('Time')
        title = (
            f'Per-Model Filtered Predictions vs Actuals ' \
            f'(first {N_days} days; {n_points} pts) ' \
            f'({representative_zone.upper()} {representative_target}, {DATASET})'
        )
        fig.suptitle(title)
        # single legend for Actual marker
        actual_handle = Line2D(
            [0], [0], marker='o', color='gray', linestyle='None',
            markersize=5, alpha=0.15
        )
        fig.legend(
            handles=[actual_handle], labels=['Actual'], loc='upper right'
        )
        # Save facet plot
        out_fname = f'filtered_facet_{representative_zone}_{representative_target}_{DATASET}.png'
        path_out = os.path.join(dirs['timeseries'], out_fname)
        fig.savefig(path_out)
        plt.close(fig)
        print('Saved facet filtered time series plot to', path_out)
        # Also generate equivalent time series plot for the full dataset over the same period
        df_ts = df_pred.copy()
        df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
        # Plot using the same start and duration on full data
        plot_time_series(df_ts, dirs, representative_zone, representative_target,
                         start_date=start_ts, days=N_days)
    except Exception as e:
        print(f'Error plotting filtered time series: {e}', file=sys.stderr)

if __name__ == '__main__':
    main()