import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from data.data_collection import get_data
import volue_insight_timeseries 
from data.curves import curve_collections

# Define the date range and set up the Volue Insight session
start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()
session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

# Define your curve names
X_curve_names = curve_collections["de"]["X"]
y_curve_names = [curve_collections["de"]["mfrr"][0]]

# Load the data
X_train, y_train, X_col, _ = get_data(X_curve_names, y_curve_names, session, start_date, end_date)

# Standardize the features. (Note: y_train is left as is, but you can also standardize or center it if appropriate.)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Split data into train and test (if not done already)
from sklearn.model_selection import train_test_split

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)



# Initialize the XGBoost regressor.
# Here we use the 'reg:squarederror' objective for regression, and some default hyperparameters.
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,      # number of boosting rounds
    learning_rate=0.01,     # step size shrinkage used in update to prevent overfitting
    max_depth=20,           # maximum depth of a tree
    random_state=42
)

# Train model on train set
xgb_model.fit(X_train_scaled, y_train.ravel())

# Evaluate on test set
test_score = xgb_model.score(X_test_scaled, y_test)
print("XGBoost model R² on Test Data:", test_score)

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# Get feature importances
feature_importances = xgb_model.feature_importances_

# Convert to DataFrame for better visualization
feat_imp_df = pd.DataFrame({'Feature': X_col, 'Importance': feature_importances})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('XGBoost Feature Importance')
plt.show()

import shap

#explainer = shap.Explainer(xgb_model, X_train_scaled)
#shap_values = explainer(X_train_scaled)
#
#shap.summary_plot(shap_values, X_train_scaled, feature_names=X_col)

from sklearn.model_selection import cross_val_score

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,      # number of boosting rounds
    learning_rate=0.01,     # step size shrinkage used in update to prevent overfitting
    max_depth=20,           # maximum depth of a tree
    random_state=42
)
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

tscv = TimeSeriesSplit(n_splits=5)  # Adjust the number of splits as needed

cv_scores = []

for train_index, test_index in tscv.split(X_train_scaled):
    X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]  

    # Train the model
    xgb_model.fit(X_train_fold, y_train_fold.ravel())
    
    # Evaluate on the test fold
    score = xgb_model.score(X_test_fold, y_test_fold)
    cv_scores.append(score)

# Print results
print("TimeSeriesSplit R² Scores:", cv_scores)
print("Mean R²:", np.mean(cv_scores))

from sklearn.feature_selection import RFE
from xgboost import XGBRegressor

# Use XGBoost as the estimator
xgb_selector = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)

# Apply Recursive Feature Elimination
rfe = RFE(xgb_selector, n_features_to_select=10)  # Adjust n_features_to_select as needed
rfe.fit(X_train_scaled, y_train)

# Print selected features
selected_features = np.array(X_col)[rfe.support_]
print("Selected Features:", selected_features)

