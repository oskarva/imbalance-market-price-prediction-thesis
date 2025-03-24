import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import dump_svmlight_file
from interpret.glassbox import ExplainableBoostingRegressor
import volue_insight_timeseries 
from data.curves import curve_collections
from data.data_collection import get_data

# ---------------------------
# Load Data and Split into Train and Test
# ---------------------------
start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()
session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

# Define curve names (using provided collections)
X_curve_names = curve_collections["de"]["X"]
y_curve_names = [curve_collections["de"]["mfrr"][0]]

# Get the data (assumed to return a DataFrame or similar structure)
X, y, X_col, _ = get_data(X_curve_names, y_curve_names, session, start_date, end_date)

# Ensure y is a 1D array
y = np.array(y).ravel()

# Optionally, scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train the Explainable Boosting Model (EBM)
# ---------------------------
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_train, y_train)

# Evaluate EBM on test data
ebm_test_score = ebm.score(X_test, y_test)
print("EBM model R² on Test Data:", ebm_test_score)

# ---------------------------
# Compute Residuals for Train and Test Sets
# ---------------------------
# Residual = actual - EBM prediction
train_predictions = ebm.predict(X_train)
residuals_train = y_train - train_predictions

test_predictions = ebm.predict(X_test)
residuals_test = y_test - test_predictions

# Verify shapes are 1D
print("train_predictions shape:", train_predictions.shape)
print("y_train shape:", y_train.shape)
print("residuals_train shape:", residuals_train.shape)

# ---------------------------
# Save Data for XGBoost External Memory Training
# ---------------------------
# Save the training data (features and residuals) in LibSVM format.
# The residuals (error) will serve as the label for XGBoost.
dump_svmlight_file(X_train, residuals_train, "train.txt", zero_based=True)
print("Training data saved to train.txt in LibSVM format.")

# Save the test data similarly (features and residuals)
dump_svmlight_file(X_test, residuals_test, "test.txt", zero_based=True)
print("Test data saved to test.txt in LibSVM format.")

# Save the EBM test predictions separately so we can combine them later.
np.save("test_ebm_pred.npy", test_predictions)
print("EBM test predictions saved to test_ebm_pred.npy.")