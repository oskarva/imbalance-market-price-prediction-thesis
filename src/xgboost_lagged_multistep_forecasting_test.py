import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

# Import data collection functions and feature engineering methods.
from data.data_collection import _get_data
from data.curves import curve_collections
import volue_insight_timeseries
from data.feature_engineering import add_time_features, add_lag_features, add_lead_features

# Define the date range and set up the Volue Insight session.
start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()
session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))

# Define curve names.
X_curve_names = curve_collections["de"]["X"]
y_curve_names = [curve_collections["de"]["mfrr"][0]]
all_curve_names = X_curve_names + y_curve_names

# Get the cleaned data.
cleaned_df = _get_data(all_curve_names, y_curve_names, session, start_date, end_date)

# Add time features.
df_features = add_time_features(cleaned_df)

# For direct forecasting, we want to predict the next 32 timesteps.
n_leads = 32
# Create lead features for the target; this yields columns like "<target>_lead1", ..., "<target>_lead32".
df_leads = add_lead_features(df_features, columns=y_curve_names, n_leads=n_leads)

# Optionally, add lag features to provide historical context.
n_lags = 32
df_final = df_leads#add_lag_features(df_leads, columns=y_curve_names, n_lags=n_lags, impute=False)

# Define the target: all 32 lead columns.
lead_cols = [f"{col}_lead{lead}" for col in y_curve_names for lead in range(1, n_leads + 1)]
# Define features as all columns except the lead targets.
feature_cols = [col for col in df_final.columns if col not in lead_cols and col not in y_curve_names]

X = df_final[feature_cols].to_numpy()
y = df_final[lead_cols].to_numpy()  # shape: (n_samples, 32)

# Standardize features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the base XGBoost regressor and wrap it with MultiOutputRegressor.
base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=20,
    random_state=42
)
model = MultiOutputRegressor(base_model)

# Train the model.
model.fit(X_train, y_train)

# Predict on the test set.
y_pred = model.predict(X_test)

# Compute metrics over all horizons (averaging the multi-output performance).
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Direct Multi-Step Forecasting Approach Metrics:")
print("Overall R²:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Plot feature importance for one horizon (e.g., for lead 1).
feat_imp = model.estimators_[0].feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': feat_imp})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('XGBoost Feature Importance (Direct Forecasting, Lead 1)')
plt.show()
