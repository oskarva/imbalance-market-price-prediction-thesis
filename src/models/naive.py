import pandas as pd
from .model_interface import ModelInterface

class Naive_Last_Known_Activation_Price(ModelInterface):

    def __init__(self):
        self.prediction = None
    
    def train(self, data:pd.DataFrame, column_name:str):
        self.prediction = self._extract_last_known_activation_price(data, column_name)

    def _extract_last_known_activation_price(self, data:pd.DataFrame, column_name:str):
        return data[column_name].iloc[-1]

    def predict(self):
        return self.prediction

    

    