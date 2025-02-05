from model_interface import ModelInterface
from sklearn.linear_model import Lasso

class Lasso(ModelInterface):

    def __init__(self, params:dict):
        if "alpha" not in params:
            raise ValueError("The parameter 'alpha' is required for Lasso")
        if "fit_intercept" not in params:
            print("Warning: 'fit_intercept' not found in params. Setting to False")
            params["fit_intercept"] = False
        self.model = Lasso(**params)
    
    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def feature_importance(self):
        return self.model.coef_, self.model.intercept_