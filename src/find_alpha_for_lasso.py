from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from data.data_collection import get_data
import volue_insight_timeseries 
import os
import pandas as pd
from data.curves import curve_collections
from sklearn.model_selection import TimeSeriesSplit

start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()
session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))
X_curve_names = curve_collections["de"]["X"]
y_curve_names = [curve_collections["de"]["mfrr"][0]]

FIT_INTERCEPT = False #(data is centered)
FOLDS = 10
RND_STATE = 42
N_ALPHAS = 100
MAX_ITER = 10000
alphas = np.logspace(-4, 2, 100)  # 0.0001 to 100

# Loading the data
X_train, y_train, X_col, _ = get_data(X_curve_names, y_curve_names, session, start_date, end_date)

# Step 1: Standardize the data (center & scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Normalize and center
y_train_scaled = y_train - y_train.mean()  # Center target variable (optional)

# Step 2: Fit the model with cross-validation

#NOTE: LassoCV uses warm_start=True by default, which means that the solution of the previous fit is used as the initial guess for the next fit. This can speed up the convergence significantly.
tscv = TimeSeriesSplit(n_splits=5)
lasso_cv = LassoCV(alphas=alphas, cv=tscv, fit_intercept=FIT_INTERCEPT, random_state=RND_STATE, max_iter=MAX_ITER)
lasso_cv.fit(X_train_scaled, y_train_scaled)

best_alpha = lasso_cv.alpha_
print(f"Optimal Alpha: {best_alpha}")
print(f"Optimal coefficient: {lasso_cv.coef_}")

for i, col in enumerate(X_col):
    print(f"{col:<30} {lasso_cv.coef_[i]}")

from sklearn.linear_model import LinearRegression

# Fit single-variable models
scores = {}
for i, col in enumerate(X_col):
    model = LinearRegression().fit(X_train_scaled[:, i].reshape(-1, 1), y_train_scaled)
    scores[col] = model.score(X_train_scaled[:, i].reshape(-1, 1), y_train_scaled)

# Sort features by R² score
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print("Top features by univariate R²:")
for col, score in sorted_scores[:10]:
    print(f"{col}: {score:.3f}")


import matplotlib.pyplot as plt

plt.scatter(X_train_scaled[:, X_col.index('pri de spot €/mwh cet h a')], y_train_scaled)
plt.xlabel('pri de spot €/mwh (scaled)')
plt.ylabel('Target (scaled)')
plt.show()

corr_matrix = pd.DataFrame(X_train_scaled, columns=X_col).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [column for column in upper.columns if any(upper[column] > 0.8)]
print("Highly correlated features:", high_corr)

X_col_filtered = [col for col in X_col if col not in high_corr]
X_train_filtered = X_train_scaled[:, [X_col.index(col) for col in X_col_filtered]]

from sklearn.linear_model import ElasticNetCV

# Try different l1_ratios (0.5 = equal L1/L2)
# Revised ElasticNet with robust settings
alphas = np.logspace(-4, 2, 100)  # 0.0001 to 100

en_cv = ElasticNetCV(
    l1_ratio=0.5,
    alphas=alphas,
    cv=TimeSeriesSplit(n_splits=5),
    max_iter=50000,
    tol=1e-3,
    random_state=42,
    n_jobs=-1
)
en_cv.fit(X_train_filtered, y_train_scaled)

print(f"ElasticNet R²: {en_cv.score(X_train_filtered, y_train_scaled):.3f}")

from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train_scaled, y_train_scaled)
print(f"Linear Regression R²: {lr.score(X_train_scaled, y_train_scaled):.3f}")