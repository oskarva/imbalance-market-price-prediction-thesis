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
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Determine start datetime for the series
    if start_date is None:
        start_dt = df['timestamp'].min()
    else:
        start_dt = pd.to_datetime(start_date)
    end_dt = start_dt + pd.Timedelta(days=days)
    mask = (df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)
    df_p = df.loc[mask]
    plt.figure(figsize=(12, 6))
    plt.plot(df_p['timestamp'], df_p['actual'], label='Actual')
    plt.plot(df_p['timestamp'], df_p['ebm_pred'], label='EBM')
    plt.plot(df_p['timestamp'], df_p['predicted'], label='Stacked')
    # Plot XGBoost and Naive predictions if available
    if 'xgb_pred' in df_p.columns:
        plt.plot(df_p['timestamp'], df_p['xgb_pred'], label='XGBoost')
    if 'naive_pred' in df_p.columns:
        plt.plot(df_p['timestamp'], df_p['naive_pred'], label='Naive')
    plt.legend()
    plt.title(f'Actual vs Predictions ({zone} {target}, {DATASET})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.tight_layout()
    # Include start date in filename to avoid overwriting
    date_str = start_dt.strftime('%Y%m%d')
    filename = f'time_series_{zone}_{target}_{date_str}_{DATASET}.png'
    plt.savefig(os.path.join(dirs['timeseries'], filename))
    plt.close()
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
    # Configurable representative case
    representative_zone = 'NO1'.lower()
    representative_target = 'up'.lower()
    TEST = "test"
    VALIDATION = "validation"
    val_or_test = VALIDATION
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
        df_pred.rename(columns={'predicted_xgb': 'xgb_pred'}, inplace=True)
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
        df_period = plot_time_series(df_pred, dirs, representative_zone, representative_target, start_date="2024-07-01", days=31)
        df_period = plot_time_series(df_pred, dirs, representative_zone, representative_target, start_date="2025-10-01", days=31)
        df_period = plot_time_series(df_pred, dirs, representative_zone, representative_target, start_date="2025-01-01", days=31)
        df_period = plot_time_series(df_pred, dirs, representative_zone, representative_target, start_date="2025-03-01", days=15)


        #plot_ebm_residuals(df_period, dirs, representative_zone, representative_target)
    print('All plots and tables saved to', base)

if __name__ == '__main__':
    main()