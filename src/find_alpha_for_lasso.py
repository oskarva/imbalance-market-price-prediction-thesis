from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np


FIT_INTERCEPT = False #(after centering data)
FOLDS = 10
RND_STATE = 42
N_ALPHAS = 100
MAX_ITER = 10000

# Step 1: Standardize the data (center & scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Normalize and center
y_train_scaled = y_train - y_train.mean()  # Center target variable (optional)

#TODO: (from sklearn docs:) "To avoid unnecessary memory duplication the X argument of the fit method should be directly passed as a Fortran-contiguous numpy array."

# Step 2: Fit the model with cross-validation

#NOTE: LassoCV uses warm_start=True by default, which means that the solution of the previous fit is used as the initial guess for the next fit. This can speed up the convergence significantly.
lasso_cv = LassoCV(n_alphas=N_ALPHAS, cv=FOLDS, fit_intercept=FIT_INTERCEPT, random_state=RND_STATE, max_iter=MAX_ITER)
lasso_cv.fit(X_train_scaled, y_train_scaled)

best_alpha = lasso_cv.alpha_
print(f"Optimal Alpha: {best_alpha}")


#Try with different amounts of folds?