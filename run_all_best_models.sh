#!/usr/bin/env bash
set -euo pipefail

 # Base data directory
 DATA_DIR=/Volumes/T9

# Python script paths
XGB_SCRIPT=src/run_cv_with_time_features.py
STACKED_SCRIPT=src/run_cv_stacked_with_time_features.py
PHASE=validation
SAMPLE=250 # REMEMBER TO UNCOMMENT LINE 57 FOR VALIDATION 

# === Stacked EBM+XGB ===
# This section uses the other script (run_cv_stacked_with_time_features.py)
# and remains unchanged from the previous version.
echo "--- Running  Stacked EBM+XGB (Best Params by Name) ---"

# Define the best XGBoost *residual* model name for each target/index combination
declare -A STACKED_RESIDUAL=(
  ["no1_0"]=xgb_residual_robust  ["no1_1"]=xgb_residual_mse
  ["no2_0"]=xgb_residual_huber   ["no2_1"]=xgb_residual_huber
  ["no3_0"]=xgb_residual_huber   ["no3_1"]=xgb_residual_mse
  ["no4_0"]=xgb_residual_huber   ["no4_1"]=xgb_residual_robust
  ["no5_0"]=xgb_residual_huber   ["no5_1"]=xgb_residual_huber
)

# Define the best EBM model name (assuming 'ebm_fast' is best for all here)
BEST_EBM_NAME="ebm_fast"

for AREA in no1 no2 no3 no4 no5; do
  for IDX in 0 1; do
    KEY="${AREA}_${IDX}"
    if [[ ! -v STACKED_RESIDUAL[$KEY] ]]; then
        echo "Warning: No best XGB residual model defined for ${KEY}. Skipping Stacked run."
        continue
    fi
    BEST_XGB_RESIDUAL_NAME=${STACKED_RESIDUAL[$KEY]}

    echo "=== Stacked EBM+XGB: ${AREA} index ${IDX} (EBM: ${BEST_EBM_NAME}, XGB: ${BEST_XGB_RESIDUAL_NAME}) ==="

    # Define output directory for this run
    OUTPUT_DIR="./results/stacked_best/${AREA}_${IDX}"
    mkdir -p ${OUTPUT_DIR} # Ensure the directory exists

    python3 ${STACKED_SCRIPT} \
      --phase ${PHASE} \
      --targets ${AREA} \
      --target-index ${IDX} \
      --run-best \
      --best-ebm-param-name ${BEST_EBM_NAME} \
      --best-xgb-param-name ${BEST_XGB_RESIDUAL_NAME} \
      --output ${OUTPUT_DIR} \
      --start 0 \
       \
      --step 1 \
      --organized-dir $DATA_DIR \
      #--sample 250 \ #uncomment this if validation
      #--no-parallel # uncomment if want to run in non-parallel mode

    # Add a small delay if needed
    # sleep 2
  done
done
echo "--- Finished Section Stacked ---"
echo


# === XGBoost‐only (Using Predefined Parameter Set Names) ===
echo "--- Running  XGBoost-only (Best Params by Name) ---"

# Define the best XGBoost parameter set name for each target/index combination
declare -A XGB_BEST_SET=(
  ["no1_0"]="set5_ensemble_diverse" ["no1_1"]="set1_robust"
  ["no2_0"]="set1_robust"           ["no2_1"]="set2_regularized"
  ["no3_0"]="set1_robust"           ["no3_1"]="set1_robust"
  ["no4_0"]="set1_robust"           ["no4_1"]="set1_robust"
  ["no5_0"]="set1_robust"           ["no5_1"]="set5_ensemble_diverse" 
)

for AREA in no1 no2 no3 no4 no5; do
  for IDX in 0 1; do
    KEY="${AREA}_${IDX}"
    if [[ ! -v XGB_BEST_SET[$KEY] ]]; then
        echo "Warning: No best XGB parameter set defined for ${KEY}. Skipping."
        continue
    fi
    BEST_PARAM_NAME=${XGB_BEST_SET[$KEY]}

    echo "=== XGBoost only: ${AREA} index ${IDX} (Set: ${BEST_PARAM_NAME}) ==="

    # Define output directory for this run
    OUTPUT_DIR="./results/xgboost_best/${AREA}_${IDX}"
    mkdir -p ${OUTPUT_DIR} # Ensure the directory exists

    python3 ${XGB_SCRIPT} \
      --phase ${PHASE} \
      --targets ${AREA} \
      --target-index ${IDX} \
      --param-set-name ${BEST_PARAM_NAME} \
      --output ${OUTPUT_DIR} \
      --organized-dir $DATA_DIR

    # Add a small delay if needed
    # sleep 1
  done
done
echo "--- Finished Section XGB ---"
echo



