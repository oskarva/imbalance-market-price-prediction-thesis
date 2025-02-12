import xgboost as xgb
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ---------------------------
# Load Data with External Memory Mode
# ---------------------------
# Append "?format=libsvm" so XGBoost knows the file format.
dtrain = xgb.DMatrix("train.txt?format=libsvm#dtrain.cache")
dtest  = xgb.DMatrix("test.txt?format=libsvm")

# ---------------------------
# Train XGBoost on the Residuals
# ---------------------------
params = {
    "objective": "reg:squarederror",
    "eta": 0.01,
    "max_depth": 20,
    "tree_method": "hist"  # 'hist' is efficient for large datasets
}
num_boost_round = 500

print("Training XGBoost model in external memory mode...")
bst = xgb.train(params, dtrain, num_boost_round)

# ---------------------------
# Make Predictions and Combine with EBM
# ---------------------------
# XGBoost predicts the residual (i.e. the correction).
xgb_correction = bst.predict(dtest)

# Load the saved EBM test predictions
test_ebm_pred = np.load("test_ebm_pred.npy")

# The labels in the test file are the residuals (i.e. y_test - ebm_test_pred),
# so we can recover y_test as:
residuals_test = dtest.get_label()
y_test = test_ebm_pred + residuals_test

# Final prediction: EBM prediction + XGBoost correction
final_predictions = test_ebm_pred + xgb_correction

# Evaluate the combined model using R²
final_r2 = r2_score(y_test, final_predictions)
print("Combined EBM + XGBoost model R² on Test Data: {:.3f}".format(final_r2))

# ---------------------------
# Plot Actual vs. Predicted Test Values
# ---------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, final_predictions, alpha=0.5, label='Predicted values')
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs. Predicted Test Values")

# Plot the ideal line where prediction == actual.
min_val = min(y_test.min(), final_predictions.min())
max_val = max(y_test.max(), final_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal')

plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# Time Series Plot: Actual vs. Predicted
# ---------------------------
# Here we assume that the test samples are in chronological order.
# If you have actual datetime values, replace 'time_index' with those values.
time_index = np.arange(len(y_test))  # This acts as a time axis

plt.figure(figsize=(10, 6))
plt.plot(time_index, y_test, label='Actual y', linewidth=2)
plt.plot(time_index, final_predictions, label='Predicted y', linestyle='--', linewidth=2)
plt.xlabel("Time Index")
plt.ylabel("y Value")
plt.title("Time Series: Actual vs. Predicted Test Values")
plt.legend()
plt.grid(True)
plt.show()