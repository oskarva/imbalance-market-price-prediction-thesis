"""
Analyze predictions from cross-validation runs to diagnose R² issues.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def load_predictions(results_dir):
    """
    Load all prediction CSV files from a results directory.
    
    Args:
        results_dir: Directory containing prediction CSVs
        
    Returns:
        Dictionary mapping round numbers to prediction DataFrames
    """
    predictions = {}
    
    # Find all prediction CSV files
    pred_files = glob.glob(os.path.join(results_dir, "*_predictions.csv"))
    
    for file_path in pred_files:
        # Extract round number from filename
        filename = os.path.basename(file_path)
        if filename.startswith("round_") and "_predictions.csv" in filename:
            round_str = filename.replace("round_", "").replace("_predictions.csv", "")
            try:
                round_num = int(round_str)
                # Load predictions
                pred_df = pd.read_csv(file_path)
                # Convert timestamp to datetime
                pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
                predictions[round_num] = pred_df
                print(f"Loaded predictions for round {round_num}")
            except ValueError:
                print(f"Could not parse round number from: {filename}")
    
    if not predictions:
        print("No prediction files found!")
    else:
        print(f"Loaded {len(predictions)} rounds of predictions")
    
    return predictions


def load_round_results(results_dir):
    """
    Load metrics for all rounds from JSON files.
    
    Args:
        results_dir: Directory containing result JSON files
        
    Returns:
        Dictionary mapping round numbers to metrics
    """
    round_results = {}
    
    # Find the most recent result file for each round
    result_files = glob.glob(os.path.join(results_dir, "round_*_results_*.json"))
    round_to_files = {}
    
    for file_path in result_files:
        filename = os.path.basename(file_path)
        parts = filename.split("_")
        if len(parts) >= 4 and parts[0] == "round" and parts[2] == "results":
            try:
                round_num = int(parts[1])
                timestamp = int(parts[3].replace(".json", ""))
                
                if round_num not in round_to_files or timestamp > round_to_files[round_num][1]:
                    round_to_files[round_num] = (file_path, timestamp)
            except (ValueError, IndexError):
                continue
    
    # Load the most recent result file for each round
    for round_num, (file_path, _) in round_to_files.items():
        try:
            with open(file_path, 'r') as f:
                round_results[round_num] = json.load(f)
            print(f"Loaded results for round {round_num}")
        except Exception as e:
            print(f"Error loading results for round {round_num}: {str(e)}")
    
    return round_results


def analyze_r2_issues(predictions, round_results):
    """
    Analyze why R² values might be extremely negative.
    
    Args:
        predictions: Dictionary of prediction DataFrames
        round_results: Dictionary of round metrics
        
    Returns:
        Dictionary of analysis results
    """
    analysis = {
        'r2_values': [],
        'variance_actual': [],
        'variance_pred': [],
        'mean_actual': [],
        'mean_pred': [],
        'range_actual': [],
        'range_pred': [],
        'rounds': []
    }
    
    for round_num, pred_df in predictions.items():
        if round_num in round_results and 'single_step' in round_results[round_num]:
            r2 = round_results[round_num]['single_step']['r2']
            
            # Calculate statistics
            var_actual = np.var(pred_df['actual'])
            var_pred = np.var(pred_df['predicted'])
            mean_actual = np.mean(pred_df['actual'])
            mean_pred = np.mean(pred_df['predicted'])
            range_actual = pred_df['actual'].max() - pred_df['actual'].min()
            range_pred = pred_df['predicted'].max() - pred_df['predicted'].min()
            
            # Store results
            analysis['r2_values'].append(r2)
            analysis['variance_actual'].append(var_actual)
            analysis['variance_pred'].append(var_pred)
            analysis['mean_actual'].append(mean_actual)
            analysis['mean_pred'].append(mean_pred)
            analysis['range_actual'].append(range_actual)
            analysis['range_pred'].append(range_pred)
            analysis['rounds'].append(round_num)
    
    # Convert to DataFrame for easier analysis
    analysis_df = pd.DataFrame(analysis)
    
    # Sort by R² value (ascending)
    analysis_df = analysis_df.sort_values('r2_values')
    
    # Add additional metrics
    analysis_df['var_ratio'] = analysis_df['variance_pred'] / analysis_df['variance_actual']
    analysis_df['mean_diff'] = analysis_df['mean_pred'] - analysis_df['mean_actual']
    analysis_df['range_ratio'] = analysis_df['range_pred'] / analysis_df['range_actual']
    
    return analysis_df


def visualize_analysis(analysis_df, predictions, output_dir):
    """
    Create visualizations to help diagnose R² issues.
    
    Args:
        analysis_df: DataFrame with analysis results
        predictions: Dictionary of prediction DataFrames
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot R² vs. Actual Variance
    plt.figure(figsize=(10, 6))
    plt.scatter(analysis_df['variance_actual'], analysis_df['r2_values'])
    plt.xscale('log')  # Often helpful for variance
    plt.xlabel('Variance of Actual Values (log scale)')
    plt.ylabel('R²')
    plt.title('R² vs. Variance of Actual Values')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "r2_vs_variance.png"))
    plt.close()
    
    # 2. Plot worst and best R² cases
    worst_rounds = analysis_df.head(3)['rounds'].tolist()
    best_rounds = analysis_df.tail(3)['rounds'].tolist()
    
    # Plot worst cases
    plt.figure(figsize=(15, 10))
    for i, round_num in enumerate(worst_rounds):
        if round_num in predictions:
            pred_df = predictions[round_num]
            r2 = analysis_df[analysis_df['rounds'] == round_num]['r2_values'].values[0]
            var_actual = analysis_df[analysis_df['rounds'] == round_num]['variance_actual'].values[0]
            
            plt.subplot(3, 2, i*2+1)
            plt.plot(pred_df['timestamp'], pred_df['actual'], 'b-', label='Actual')
            plt.plot(pred_df['timestamp'], pred_df['predicted'], 'r--', label='Predicted')
            plt.title(f'Round {round_num} (R²={r2:.4f}, Var={var_actual:.4f})')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 2, i*2+2)
            plt.scatter(pred_df['actual'], pred_df['predicted'])
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            
            # Add perfect prediction line
            min_val = min(pred_df['actual'].min(), pred_df['predicted'].min())
            max_val = max(pred_df['actual'].max(), pred_df['predicted'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "worst_r2_cases.png"))
    plt.close()
    
    # Plot best cases
    plt.figure(figsize=(15, 10))
    for i, round_num in enumerate(best_rounds):
        if round_num in predictions:
            pred_df = predictions[round_num]
            r2 = analysis_df[analysis_df['rounds'] == round_num]['r2_values'].values[0]
            var_actual = analysis_df[analysis_df['rounds'] == round_num]['variance_actual'].values[0]
            
            plt.subplot(3, 2, i*2+1)
            plt.plot(pred_df['timestamp'], pred_df['actual'], 'b-', label='Actual')
            plt.plot(pred_df['timestamp'], pred_df['predicted'], 'r--', label='Predicted')
            plt.title(f'Round {round_num} (R²={r2:.4f}, Var={var_actual:.4f})')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 2, i*2+2)
            plt.scatter(pred_df['actual'], pred_df['predicted'])
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            
            # Add perfect prediction line
            min_val = min(pred_df['actual'].min(), pred_df['predicted'].min())
            max_val = max(pred_df['actual'].max(), pred_df['predicted'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5)
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_r2_cases.png"))
    plt.close()
    
    # 3. Distribution of key metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(analysis_df['r2_values'])
    plt.xlabel('R²')
    plt.title('Distribution of R² Values')
    
    plt.subplot(2, 2, 2)
    sns.histplot(analysis_df['variance_actual'])
    plt.xlabel('Variance of Actual Values')
    plt.title('Distribution of Actual Variance')
    
    plt.subplot(2, 2, 3)
    sns.histplot(analysis_df['var_ratio'].clip(0, 10))  # Clip to reasonable range
    plt.xlabel('Variance Ratio (Predicted/Actual)')
    plt.title('Distribution of Variance Ratio')
    
    plt.subplot(2, 2, 4)
    sns.histplot(analysis_df['mean_diff'])
    plt.xlabel('Mean Difference (Predicted - Actual)')
    plt.title('Distribution of Mean Difference')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_distributions.png"))
    plt.close()
    
    # 4. Correlation matrix
    corr_cols = ['r2_values', 'variance_actual', 'variance_pred', 
                'mean_actual', 'mean_pred', 'var_ratio', 'mean_diff']
    corr_matrix = analysis_df[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_correlations.png"))
    plt.close()
    
    # 5. Save analysis table
    analysis_df.to_csv(os.path.join(output_dir, "r2_analysis.csv"), index=False)
    
    # Return most important findings
    findings = {
        'worst_r2': analysis_df['r2_values'].min(),
        'best_r2': analysis_df['r2_values'].max(),
        'median_r2': analysis_df['r2_values'].median(),
        'min_variance_actual': analysis_df['variance_actual'].min(),
        'correlation_var_r2': corr_matrix.loc['variance_actual', 'r2_values'],
        'rounds_with_zero_variance': (analysis_df['variance_actual'] < 0.0001).sum()
    }
    
    return findings

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze prediction results and diagnose R² issues')
    parser.add_argument('--results_dir', type=str, required=True, 
                        help='Directory containing cross-validation results')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Directory to save analysis results and visualizations')
    
    args = parser.parse_args()
    
    print(f"Analyzing results from: {args.results_dir}")
    
    # Load predictions and results
    predictions = load_predictions(args.results_dir)
    round_results = load_round_results(args.results_dir)
    
    if not predictions or not round_results:
        print("No data to analyze!")
        return
    
    # Analyze R² issues
    print("Analyzing R² issues...")
    analysis_df = analyze_r2_issues(predictions, round_results)
    
    # Create visualizations
    print("Creating visualizations...")
    findings = visualize_analysis(analysis_df, predictions, args.output_dir)
    
    # Print summary findings
    print("\nAnalysis Summary:")
    print(f"Worst R²: {findings['worst_r2']:.4f}")
    print(f"Best R²: {findings['best_r2']:.4f}")
    print(f"Median R²: {findings['median_r2']:.4f}")
    print(f"Minimum variance in actual values: {findings['min_variance_actual']:.8f}")
    print(f"Correlation between variance and R²: {findings['correlation_var_r2']:.4f}")
    print(f"Rounds with near-zero variance: {findings['rounds_with_zero_variance']}")
    
    # Generate recommendations based on findings
    print("\nRecommendations:")
    
    if findings['rounds_with_zero_variance'] > 0:
        print("- Some test sets have near-zero variance, which can lead to unstable R² values.")
        print("  Consider excluding these rounds from your R² calculations.")
    
    if findings['correlation_var_r2'] > 0.3:
        print("- There's a correlation between variance and R². Consider normalizing your target")
        print("  variable or using a weighted R² calculation.")
    
    if findings['worst_r2'] < -100:
        print("- Extremely negative R² values suggest your model may be worse than a")
        print("  constant predictor. Check for data leakage or scaling issues.")
    
    print("\nAnalysis results saved to:", args.output_dir)


if __name__ == "__main__":
    main()