import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# Import your data collection functions and feature engineering methods.
from data.data_collection import _get_data  # Using the internal function to retrieve the cleaned DataFrame.
from data.curves import curve_collections
import volue_insight_timeseries
from data.feature_engineering import prepare_features

# Define the date range and set up the Volue Insight session.
start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()
session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

# Define your curve names.
X_curve_names = curve_collections["de"]["X"]
y_curve_names = [curve_collections["de"]["mfrr"][0]]
all_curve_names = X_curve_names + y_curve_names

# Get the cleaned data (as a DataFrame with a DatetimeIndex).
cleaned_df = _get_data(all_curve_names, y_curve_names, session, start_date, end_date)

# Use feature engineering: add time features and lag features (n_lags=32) for the target.
n_lags = 32
df_features = prepare_features(cleaned_df, target_columns=y_curve_names, n_lags=n_lags, include_time=True)

# Define the features and target.
# Here, the target is the current (non-shifted) value; features include the lagged y and time features.
feature_cols = [col for col in df_features.columns if col not in y_curve_names]
target_cols = y_curve_names

X = df_features[feature_cols].to_numpy()
y = df_features[target_cols].to_numpy()

# Standardize the features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost regressor.
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=20,
    random_state=42
)
xgb_model.fit(X_train, y_train.ravel())

# Make predictions and evaluate.
y_pred = xgb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Recursive Forecasting Approach Metrics:")
print("R²:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Plot feature importances.
feature_importances = xgb_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': feature_importances})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('XGBoost Feature Importance (Recursive Forecasting)')
plt.show()
