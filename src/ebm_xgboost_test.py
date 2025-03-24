import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from data.data_collection import get_data
import volue_insight_timeseries 
from data.curves import curve_collections
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import train_test_split
import resource
import resource

import resource

# Define 4GB in bytes
four_gb = 4 * 1024 ** 3

# Get current limits (optional, for information)
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
print("Current soft limit:", soft)
print("Current hard limit:", hard)

# Set the new limit to 4GB for both soft and (if allowed) hard limits
# Note: On some systems, one might not be allowed to set the hard limit lower than it is.
try:
    resource.setrlimit(resource.RLIMIT_AS, (four_gb, hard))
    print("Memory limit set to 4GB.")
except ValueError as e:
    print("Error setting memory limit:", e)
except resource.error as e:
    print("Resource error:", e)

# Define the date range and set up the Volue Insight session
start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp.today()
session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

# Define curve names
X_curve_names = curve_collections["de"]["X"]
y_curve_names = [curve_collections["de"]["mfrr"][0]]

# Load the data
X_train, y_train, X_col, _ = get_data(X_curve_names, y_curve_names, session, start_date, end_date)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


ebm = ExplainableBoostingRegressor(
    random_state=42,
)

ebm.fit(X_train, y_train)

test_score = ebm.score(X_test, y_test)
print("EBM model R² on Test Data:", test_score)

# Now, lets train an xgboost model on the residuals of the EBM model
residuals = y_train - ebm.predict(X_train)
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,      # number of boosting rounds
    learning_rate=0.1,     # step size shrinkage used in update to prevent overfitting
    max_depth=8,           # maximum depth of a tree
    random_state=42
)

# Train model on train set
xgb_model.fit(X_train, residuals)

# Evaluate on test set
test_score = xgb_model.score(X_test, y_test)
print("XGBoost model R² on Test Data:", test_score)