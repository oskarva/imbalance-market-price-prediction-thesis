class ModelInterface:
    def __init__(self, model):
        self.model = model

    def train(self, X, y):
        return NotImplementedError

    def predict(self, X):
        return NotImplementedError