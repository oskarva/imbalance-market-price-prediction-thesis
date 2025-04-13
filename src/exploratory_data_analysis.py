import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import re
import glob
import argparse
import warnings
import traceback

# Ignore harmless warnings
warnings.filterwarnings("ignore", message="A date index has been provided")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform EDA on training data from expanding window cross-validation.")
    parser.add_argument(
        "--data-folder",
        type=str,
        default="/Volumes/T9", # Default base path
        help="Path to the base folder containing area subfolders (e.g., /Volumes/T9)."
    )
    parser.add_argument(
        "--areas",
        nargs='+', # Allows multiple areas
        default=["no1", "no2", "no3", "no4", "no5"], # Default to all areas
        help="List of area names to process (e.g., no1 no2)."
    )
    return parser.parse_args()

# --- Helper Functions ---
def find_last_round_files(validation_folder):
    """Finds the X_train and y_train files for the highest round number."""
    x_files = glob.glob(os.path.join(validation_folder, "X_train_*.csv"))
    y_files = glob.glob(os.path.join(validation_folder, "y_train_*.csv"))

    if not x_files or not y_files:
        print(f"Warning: No X_train or y_train files found in {validation_folder}")
        return None, None, -1

    max_round = -1
    last_x_file = None
    last_y_file = None

    # Find max round number from X files
    for f in x_files:
        match = re.search(r'X_train_(\d+)\.csv$', os.path.basename(f))
        if match:
            round_num = int(match.group(1))
            max_round = max(max_round, round_num)

    if max_round == -1:
         print(f"Warning: Could not parse round number from X_train files in {validation_folder}")
         return None, None, -1

    last_x_file = os.path.join(validation_folder, f"X_train_{max_round}.csv")
    last_y_file = os.path.join(validation_folder, f"y_train_{max_round}.csv")

    if not os.path.exists(last_x_file):
        print(f"Warning: Expected file {last_x_file} does not exist.")
        last_x_file = None
    if not os.path.exists(last_y_file):
        print(f"Warning: Expected file {last_y_file} does not exist.")
        last_y_file = None

    return last_x_file, last_y_file, max_round



def load_and_prepare_data(x_path, y_path):
    """Loads X and y CSVs, merges them, sets DatetimeIndex."""
    try:
        print(f"Attempting to read X_train: {x_path}")
        df_x = pd.read_csv(x_path, index_col=0)
        print(f"Successfully read X_train. Shape: {df_x.shape}, Index Name: {df_x.index.name}, Head:\n{df_x.head(2)}")

        print(f"\nAttempting to read y_train: {y_path}")
        df_y = pd.read_csv(y_path, index_col=0)
        print(f"Successfully read y_train. Shape: {df_y.shape}, Index Name: {df_y.index.name}, Head:\n{df_y.head(2)}")

        print("\nAttempting datetime conversion for X_train index...")
        # Use errors='coerce' to handle potential parsing issues gracefully
        df_x.index = pd.to_datetime(df_x.index, utc=True, errors='coerce').tz_convert('Europe/Oslo')
        print(f"X_train datetime conversion done. Index head:\n{df_x.index[:5]}")
        
        print("Checking for NaT in X_train index after conversion...")
        original_x_len = len(df_x)
        df_x = df_x[pd.notna(df_x.index)] # Keep rows where index is NOT NaT
        print(f"X_train shape after removing NaT indices: {df_x.shape} (removed {original_x_len - len(df_x)} rows)")
        

        print("\nAttempting datetime conversion for y_train index...")
        df_y.index = pd.to_datetime(df_y.index, utc=True, errors='coerce').tz_convert('Europe/Oslo')
        print(f"y_train datetime conversion done. Index head:\n{df_y.index[:5]}")
        
        print("Checking for NaT in y_train index after conversion...")
        original_y_len = len(df_y)
        df_y = df_y[pd.notna(df_y.index)] # Keep rows where index is NOT NaT
        print(f"y_train shape after removing NaT indices: {df_y.shape} (removed {original_y_len - len(df_y)} rows)")
        

        if df_x.empty or df_y.empty:
             print("Warning: One or both DataFrames became empty after datetime conversion and NaT removal.")
             # Optional: Decide whether to return None or empty DataFrame
             # return None # Or return an empty df_merged if needed downstream

        print("\nAttempting to merge X and y DataFrames...")
        # Ensure indices are named the same or merge might fail if one df became empty
        # Though merge on index should work even with unnamed indices if they are DatetimeIndex
        df_merged = pd.merge(df_x, df_y, left_index=True, right_index=True, how='inner')
        print(f"Merge successful. Merged shape: {df_merged.shape}")

        if df_merged.empty:
            print("Warning: Merged DataFrame is empty. Check index alignment and NaNs in original files.")

        print(f"Data loaded and prepared successfully.")
        return df_merged

    except FileNotFoundError:
        print(f"Error: One or both files not found: {x_path}, {y_path}")
        return None
    except Exception as e:
        print(f"\n--- Error during data loading/preparation ---")
        print(f"Error Type: {type(e)}")
        print(f"Error Message: {e}")
        print("Traceback:")
        traceback.print_exc() # Print the full traceback
        print("--- End Traceback ---")
        return None


# --- EDA Function (with NO5 wind exception) ---
def perform_eda(df_eda, area):
    """Performs and plots EDA for the given DataFrame and area."""
    print(f"\n--- Starting EDA for Area: {area} ---")

    # Define column names dynamically for the current area
    target_up_col = f"pri {area} regulation up €/mwh cet min15 a"
    target_down_col = f"pri {area} regulation down €/mwh cet min15 a"
    consumption_actual_col = f"con {area} intraday mwh/h cet h a"
    spot_price_col = f"pri {area} spot €/mwh cet h a"
    hydro_actual_col = f"pro {area} hydro tot mwh/h cet h af"
    wind_actual_col = f"pro {area} wnd mwh/h cet min15 a" # Wind actual
    rdl_actual_col = f"rdl {area} mwh/h cet min15 a"
    heating_actual_col = f"con {area} heating % cet min15 s"
    cooling_actual_col = f"con {area} cooling % cet min15 s"
    # Forecast column names are no longer needed for this specific EDA script

    # Base list of features present in training data
    feature_cols_eda = [
        consumption_actual_col, spot_price_col, hydro_actual_col,
        wind_actual_col, rdl_actual_col, heating_actual_col, cooling_actual_col
    ]

    # ---  NO5 EXCEPTION for WIND  ---
    if area == 'no5':
        print(f"Note: Excluding wind columns for area {area} as data is unavailable.")
        if wind_actual_col in feature_cols_eda:
             feature_cols_eda.remove(wind_actual_col)

    # Combine targets and features
    all_cols_eda = [target_up_col, target_down_col] + feature_cols_eda
    # Filter based on columns actually present in the loaded DataFrame
    all_cols_eda = [col for col in all_cols_eda if col in df_eda.columns]
    feature_cols_eda = [col for col in feature_cols_eda if col in df_eda.columns]

    if not all_cols_eda:
        print(f"Error: No expected columns found for EDA in area {area}.")
        return

    # Add time components if needed
    if 'hour' not in df_eda.columns: df_eda['hour'] = df_eda.index.hour
    if 'dayofweek' not in df_eda.columns: df_eda['dayofweek'] = df_eda.index.dayofweek
    if 'month' not in df_eda.columns: df_eda['month'] = df_eda.index.month

    # --- 1. Basic Info & Stats ---
    print(f"\n--- 1. Basic Info & Stats ({area}) ---")
    print(f"Summary Statistics for {area}:")
    print(df_eda[all_cols_eda].describe().round(2))

    # --- 2. Target Variable Time Series Plots ---
    print(f"\n--- 2. Plotting Target Variables ({area}) ---")
    # Plot UP price
    if target_up_col in df_eda.columns:
        plt.figure(figsize=(15, 6))
        plt.plot(df_eda.index, df_eda[target_up_col], label=f'{target_up_col}', linewidth=0.8)
        plt.title(f'Time Series: {target_up_col}')
        plt.xlabel("Time"); plt.ylabel("Price (€/MWh)"); plt.legend(); plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(df_eda.index.year.unique()) // 4)))
        plt.gcf().autofmt_xdate()
        plt.suptitle(f"EDA Plots for Area: {area}", fontsize=16, y=1.03) # Keep overall title for first plot
    else: print(f"Column not found: {target_up_col}")

    # Plot DOWN price
    if target_down_col in df_eda.columns:
        plt.figure(figsize=(15, 6))
        plt.plot(df_eda.index, df_eda[target_down_col], label=f'{target_down_col}', linewidth=0.8, color='orange')
        plt.title(f'Time Series: {target_down_col}')
        plt.xlabel("Time"); plt.ylabel("Price (€/MWh)"); plt.legend(); plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(df_eda.index.year.unique()) // 4)))
        plt.gcf().autofmt_xdate()
    else: print(f"Column not found: {target_down_col}")

    # --- 3. Target Variable ACF and PACF Plots ---
    print(f"\n--- 3. Plotting ACF/PACF ({area}) ---")
    # Plot for UP price
    if target_up_col in df_eda.columns:
        target_series_up = df_eda[target_up_col].dropna()
        if not target_series_up.empty:
            fig_acf_up, axes_acf_up = plt.subplots(1, 2, figsize=(16, 5))
            lags_to_plot = min(96 * 2, len(target_series_up)//2 - 1)
            if lags_to_plot > 1:
                 plot_acf(target_series_up, lags=lags_to_plot, ax=axes_acf_up[0], title=f'ACF - {target_up_col}')
                 plot_pacf(target_series_up, lags=lags_to_plot, ax=axes_acf_up[1], title=f'PACF - {target_up_col}', method='ywm')
                 plt.tight_layout()
            else: print(f"Not enough data points for ACF/PACF plot for {target_up_col}")
        else: print(f"No data after dropping NaNs for ACF/PACF plot: {target_up_col}")
    else: print(f"Column not found: {target_up_col}")

    # Plot for DOWN price
    if target_down_col in df_eda.columns:
        target_series_down = df_eda[target_down_col].dropna()
        if not target_series_down.empty:
            fig_acf_down, axes_acf_down = plt.subplots(1, 2, figsize=(16, 5))
            lags_to_plot = min(96 * 2, len(target_series_down)//2 - 1)
            if lags_to_plot > 1:
                 plot_acf(target_series_down, lags=lags_to_plot, ax=axes_acf_down[0], title=f'ACF - {target_down_col}')
                 plot_pacf(target_series_down, lags=lags_to_plot, ax=axes_acf_down[1], title=f'PACF - {target_down_col}', method='ywm')
                 plt.tight_layout()
            else: print(f"Not enough data points for ACF/PACF plot for {target_down_col}")
        else: print(f"No data after dropping NaNs for ACF/PACF plot: {target_down_col}")
    else: print(f"Column not found: {target_down_col}")


    # --- 4. Seasonality Analysis ---
    print(f"\n--- 4. Seasonality Analysis ({area}) ---")
    # Analysis for UP price
    if target_up_col in df_eda.columns:
        print(f"--- Seasonality Analysis for {target_up_col} ---")
        fig_seas_up, axes_seas_up = plt.subplots(3, 1, figsize=(12, 15))
        sns.boxplot(x='hour', y=target_up_col, data=df_eda, ax=axes_seas_up[0])
        axes_seas_up[0].set_title(f'Seasonality by Hour - {target_up_col}'); axes_seas_up[0].grid(True, axis='y')
        sns.boxplot(x='dayofweek', y=target_up_col, data=df_eda, ax=axes_seas_up[1])
        axes_seas_up[1].set_title(f'Seasonality by Day of Week - {target_up_col}'); axes_seas_up[1].set_xlabel("Day of Week (0=Mon, 6=Sun)"); axes_seas_up[1].grid(True, axis='y')
        sns.boxplot(x='month', y=target_up_col, data=df_eda, ax=axes_seas_up[2])
        axes_seas_up[2].set_title(f'Seasonality by Month - {target_up_col}'); axes_seas_up[2].grid(True, axis='y')
        plt.tight_layout()
        plt.suptitle(f"Seasonality: {target_up_col}", y=1.02)

        print(f"Performing Seasonal Decomposition (Daily) for {target_up_col}")
        decomposition_series = df_eda[target_up_col].dropna().last('365D')
        if len(decomposition_series) >= 96*2:
            try:
                result = seasonal_decompose(decomposition_series, model='additive', period=96, extrapolate_trend='freq')
                fig_decomp = result.plot()
                fig_decomp.set_size_inches((14, 10))
                fig_decomp.suptitle(f'Seasonal Decomposition (Daily) - {target_up_col}', y=1.01)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            except ValueError as ve: print(f"Could not perform seasonal decomposition: {ve}")
            except Exception as e: print(f"Could not perform seasonal decomposition: {e}")
        else: print("Not enough data for daily seasonal decomposition plot.")
    else: print(f"Column not found for Seasonality Analysis: {target_up_col}")

    # Analysis for DOWN price
    if target_down_col in df_eda.columns:
        print(f"\n--- Seasonality Analysis for {target_down_col} ---")
        fig_seas_down, axes_seas_down = plt.subplots(3, 1, figsize=(12, 15))
        sns.boxplot(x='hour', y=target_down_col, data=df_eda, ax=axes_seas_down[0])
        axes_seas_down[0].set_title(f'Seasonality by Hour - {target_down_col}'); axes_seas_down[0].grid(True, axis='y')
        sns.boxplot(x='dayofweek', y=target_down_col, data=df_eda, ax=axes_seas_down[1])
        axes_seas_down[1].set_title(f'Seasonality by Day of Week - {target_down_col}'); axes_seas_down[1].set_xlabel("Day of Week (0=Mon, 6=Sun)"); axes_seas_down[1].grid(True, axis='y')
        sns.boxplot(x='month', y=target_down_col, data=df_eda, ax=axes_seas_down[2])
        axes_seas_down[2].set_title(f'Seasonality by Month - {target_down_col}'); axes_seas_down[2].grid(True, axis='y')
        plt.tight_layout()
        plt.suptitle(f"Seasonality: {target_down_col}", y=1.02)

        print(f"Performing Seasonal Decomposition (Daily) for {target_down_col}")
        decomposition_series_down = df_eda[target_down_col].dropna().last('365D')
        if len(decomposition_series_down) >= 96*2:
            try:
                result_down = seasonal_decompose(decomposition_series_down, model='additive', period=96, extrapolate_trend='freq')
                fig_decomp_down = result_down.plot()
                fig_decomp_down.set_size_inches((14, 10))
                fig_decomp_down.suptitle(f'Seasonal Decomposition (Daily) - {target_down_col}', y=1.01)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            except ValueError as ve: print(f"Could not perform seasonal decomposition: {ve}")
            except Exception as e: print(f"Could not perform seasonal decomposition: {e}")
        else: print("Not enough data for daily seasonal decomposition plot.")
    else: print(f"Column not found for Seasonality Analysis: {target_down_col}")


    # --- 5. Feature Correlation Heatmap ---
    print(f"\n--- 5. Plotting Feature Correlation Heatmap ({area}) ---")
    # Use all_cols_eda which includes both targets if present
    numeric_cols_for_corr = df_eda[all_cols_eda].select_dtypes(include=np.number).columns
    if len(numeric_cols_for_corr) > 1:
        numeric_cols_for_corr = [col for col in numeric_cols_for_corr if col not in ['hour', 'dayofweek', 'month']]
        if len(numeric_cols_for_corr) > 1:
            corr_matrix = df_eda[numeric_cols_for_corr].corr()
            plt.figure(figsize=(10, 8)) # Adjusted size
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 7})
            plt.title(f'Correlation Matrix ({area})')
            plt.xticks(rotation=45, ha='right', fontsize=8); plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
        else: print("Not enough numeric, non-time columns for correlation heatmap.")
    else: print("Not enough numeric columns for correlation heatmap.")

    print(f"\n--- Finished EDA for Area: {area} ---")


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_arguments()
    base_folder = args.data_folder
    areas_to_process = args.areas

    print(f"Starting EDA for areas: {areas_to_process}")
    print(f"Base data folder: {base_folder}")

    # Make plot showing interactive
    plt.ion() # Turn interactive mode on

    for area in areas_to_process:
        print(f"\nProcessing Area: {area}")
        area_folder = os.path.join(base_folder, area)
        validation_folder = os.path.join(area_folder, "validation_rounds")

        if not os.path.isdir(validation_folder):
            print(f"Error: Validation folder not found: {validation_folder}")
            continue

        x_file, y_file, last_round = find_last_round_files(validation_folder)

        if x_file and y_file:
            print(f"Found last round {last_round} files for {area}.")
            df_area_eda = load_and_prepare_data(x_file, y_file)

            if df_area_eda is not None and not df_area_eda.empty:
                perform_eda(df_area_eda, area)
                 # Show plots per area, or collect handles and show all at end
                plt.show(block=False) # Show plot, but allow script to continue
                plt.pause(1) # Pause briefly to allow plot rendering
            else:
                print(f"Skipping EDA for {area} due to data loading error or empty dataframe.")
        else:
            print(f"Skipping EDA for {area}: Could not find valid last round files.")

    print("\n--- All EDA Processing Complete ---")
    plt.ioff() # Turn interactive mode off
    plt.show() # Keep final plots open