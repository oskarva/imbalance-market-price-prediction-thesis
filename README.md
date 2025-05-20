# mFRR Activation Price Forecasting for the Norwegian Imbalance Energy Market

This repository contains the Python code and experiments supporting the master’s thesis:
“mFRR Activation Price Forecasting for the Norwegian Imbalance Energy Market: Investigating the Tradeoff between Accuracy and Model Interpretability”
by Oskar Våle (2025). \
**If you have any questions at all, feel free to reach out to me at oskar.vaale@gmail.com. Do not hesitate to fork this repo and make it your own:)**

<!-- TOC -->
- [Project Overview](#project-overview)
- [Problem Domain](#problem-domain)
- [Methodology](#methodology)
- [Key Findings and Insights](#key-findings-and-insights)
- [Repository Contents](#repository-contents)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Outputs](#outputs)
- [Limitations](#limitations)
- [Citation](#citation)
<!-- /TOC -->

## Project Overview

**Title:**  
mFRR Activation Price Forecasting for the Norwegian Imbalance Energy Market: Investigating the Tradeoff between Accuracy and Model Interpretability

**Author:**  
Oskar Våle

**Abstract:**  
This thesis focuses on forecasting 8-hour ahead Manual Frequency Restoration Reserve (mFRR) activation prices in Norway’s five price zones (NO1–NO5). It investigates the trade-off between predictive accuracy and model interpretability for this highly volatile and challenging forecast task.

**Main Research Objective:**  
Accurately predict mFRR activation prices while ensuring high model interpretability and deriving insights into the price dynamics.

**Specific Aims:**  
- Assess the performance of XGBoost for 8-hour ahead mFRR price forecasting.  
- Evaluate the Explainable Boosting Machine (EBM) to analyze the accuracy-interpretability trade-off.  
- Implement and evaluate a residual-based stacked ensemble combining XGBoost and EBM.  
- Leverage EBM’s transparency to understand key price drivers across regions.

## Problem Domain

The Norwegian Imbalance Energy Market handles real-time discrepancies between electricity supply and demand using activation prices for manual frequency restoration reserves (mFRR). These prices are settled in the mFRR energy activation market based on a marginal pricing principle and come in upward (power deficit) and downward (power surplus) variants. A significant amount of the activation prices equal the day-ahead spot price, but the remainder can deviate significantly, posing forecasting challenges.

Key characteristics:  
- Five price zones: NO1, NO2, NO3, NO4, NO5.  
- High volatility and regime changes, especially during price deviations from the day-ahead spot.  
- Crucial for market participants and system operators to manage balancing costs.

## Methodology

- **Data Source:** Time series data with 15-minute resolution (Jan 2021–Mar 2025) via the Volue Insight API.  
- **Forecast Horizon:** 8 hours (32 intervals of 15 min).  
- **Price Zones:** NO1–NO5.  
- **Models:**  
  - Naïve baseline model  
  - XGBoost (eXtreme Gradient Boosting)  
  - Explainable Boosting Machine (EBM)  
  - Stacked ensemble (EBM + XGBoost on residuals)  
- **Features:** Day-ahead spot price, consumption, hydropower and wind production, residual load, heating/cooling needs, cyclic time features (hour, month).  
- **Evaluation:**  
  - Expanding-window cross-validation through time  
  - Metrics: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), R² (Coefficient of Determination)  
- **Tools & Libraries:** Python 3.9.6; Pandas, NumPy, Scikit-learn, XGBoost, Interpret (EBM), Matplotlib, Joblib, Statsmodels, Tabulate.

## Key files
The most important files with regards to implementation details of the models are the following:
   - Script that trains and tests the naïve baseline model using cross-validation: `python src/run_naive_cv_all_targets.py`  
   - Script that trains and tests the XGBoost standalone model using cross-validation: `python src/run_cv_with_time_features.py`  
   - Script that trains and tests the EBM and XGBoost stacked ensemble using cross-validation: `python src/run_cv_stacked_with_time_features.py`

## Key Findings and Insights

- Both XGBoost and EBM outperform the naïve baseline by large margins, demonstrating the value of gradient-boosting approaches.  
- The EBM achieves predictive accuracy on par with XGBoost across most zones, enabling interpretability with negligible performance loss.  
- The residual-based stacked model yields marginal improvements, suggesting alternative stacking strategies (e.g., classification-then-regression) may be more effective.  
- Forecasting periods with activation price deviations from the day-ahead spot remains challenging; model performance drops when isolating these events.  
- EBM shape functions reveal the dominant influence of the spot price and non-linear, region-specific effects of drivers such as hydropower output and residual load—particularly in NO3 and NO4.

## Repository Contents

```
.        
├── chapters/                   # LaTeX source for thesis chapters
├── models/                     # Serialized trained model files
├── optimization_runs/          # Hyperparameter search logs and results
├── plots/                      # Exploratory and results plots
├── results/                    # Forecast results and evaluation metrics
├── src/                        # Python source code
│   └── data/                   # Data loading & preprocessing utilities
├── run_all_best_models.sh      # Bash script to run training & evaluation with best parameters
├── requirements.txt            # Python package dependencies
└── README.md                   # This file
```

## Getting Started

**Prerequisites:**  
- Python 3.9.6 (or later)  
- Access to the Volue Insight API  
- UNIX-like shell for running the provided `.sh` scripts

**Installation:**  
```bash
git clone https://github.com/oskarva/imbalance-market-price-prediction-thesis.git
cd imbalance-market-price-prediction-thesis
pip install --upgrade pip
pip install -r requirements.txt
```

**API Configuration:**  
Set the `WAPI_CONFIG` environment variable to point to your Volue Insight API credentials file:
```bash
export WAPI_CONFIG=/path/to/volue_api_config.ini
```
Failure to configure this will prevent data collection via `src/collect_data.py`. \
*Important:* This repository **does not contain the raw dataset** used for the thesis, due to its size and proprietary nature. To reproduce the experiments or run the code on the original data period (Jan 2021–Mar 2025), you will need to fetch the data using `src/collect_data.py` with your own API access **OR** collect the data through other means, saving the data in a format compatible with the rest of my scripts. 


## Usage

1. **Data Collection & Preparation**  
   Generate two-stage rolling cross-validation splits in `src/data/csv`:  
   ```bash
   python src/collect_data.py \
     --start-date 2021-01-01 \
     --val-start-date 2023-12-16 \
     --test-start-date 2024-06-16 \
     --end-date 2025-03-15 \
     --output-dir src/data/csv
   ```
2. **Model Training & Evaluation**  
   Run all experiments with best parameter sets (as defined in `run_all_best_models.sh`):  
   ```bash
   bash run_all_best_models.sh
   ```
3. **Individual Scripts:**  
   - Naïve baseline: `python src/run_naive_cv_all_targets.py`  
   - XGBoost CV: `python src/run_cv_with_time_features.py`  
   - Stacked ensemble: `python src/run_cv_stacked_with_time_features.py`

## Outputs

- **Results:** Stored under `results/`, organized by model type and area/index.  
- **Plots:** Forecast vs. actual, EBM shape functions, performance comparisons in `plots/`.  
- **Metrics:** MAE, RMSE, and R² tables for each zone and model.

## Hyperparameters 

The best found hyperparameter sets for each area and model can be found in `run_all_best_models.sh`. \
For any given model, the different hyperparameter sets tried will be hardcoded into the script for that model. \
So, to find the parametersets tried for the standalone XGBoost, go to `src/run_cv_with_time_features.py` and Ctrl + F "`PREDEFINED_PARAMETER_SETS`" or the name of a specific parameter set found in `run_all_best_models.sh`. \
To find the parameter sets tried for the EBM, go to `src/run_cv_stacked_with_time_features.py` and Ctrl + F "`ebm_parameter_sets`" or a specific parameter set name. \
To find the parameter sets tried for the XGBoost residual model, go to `src/run_cv_stacked_with_time_features.py` and Ctrl + F "`xgb_residual_sets`"or a specific parameter set name.

## Limitations

- Models struggle to predict activation price deviations from the day-ahead spot: Current features may not capture the underlying triggers.  
- Hyperparameter tuning was limited to predefined search spaces; alternative strategies may yield improvements.  
- The stacked ensemble shows minimal gains; other stacking or hybrid approaches warrant exploration.  
- Reproducibility depends on access to the Volue Insight API and full data coverage (Jan 2021–Mar 2025).

## Citation

If you use this code or results, please cite:
> Våle, O. (2025). mFRR Activation Price Forecasting for the Norwegian Imbalance Energy Market: Investigating the Tradeoff between Accuracy and Model Interpretability. Master’s Thesis, University of Oslo (UIO).  
>
> GitHub: https://github.com/oskarva/imbalance-market-price-prediction-thesis