from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from data.data_collection import get_data
import volue_insight_timeseries 
import os
import pandas as pd
from data.curves import curve_collections

start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()
session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))
X_curve_names = curve_collections["de"]["X"]
y_curve_names = [curve_collections["de"]["mfrr"][1]]

FIT_INTERCEPT = False #(data is centered)
FOLDS = 10
RND_STATE = 42
N_ALPHAS = 100
MAX_ITER = 10000

# Loading the data
X_train, y_train = get_data(X_curve_names, y_curve_names, session, start_date, end_date)

# Step 1: Standardize the data (center & scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Normalize and center
y_train_scaled = y_train - y_train.mean()  # Center target variable (optional)

# Step 2: Fit the model with cross-validation

#NOTE: LassoCV uses warm_start=True by default, which means that the solution of the previous fit is used as the initial guess for the next fit. This can speed up the convergence significantly.
lasso_cv = LassoCV(n_alphas=N_ALPHAS, cv=FOLDS, fit_intercept=FIT_INTERCEPT, random_state=RND_STATE, max_iter=MAX_ITER)
lasso_cv.fit(X_train_scaled, y_train_scaled)

best_alpha = lasso_cv.alpha_
print(f"Optimal Alpha: {best_alpha}")
print(f"Optimal coefficient: {lasso_cv.coef_}")

for x in X_curve_names:
    print(f"{x:<30} {lasso_cv.coef_[X_curve_names.index(x)]}")

#Try with different amounts of folds?