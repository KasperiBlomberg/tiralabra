import numpy as np
from src.utils.activation_functions import ReLU, softmax


class NeuralNetwork:
    def __init__(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2

    def forward_prop(self, X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = ReLU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def predict(self, X):
        """Predicts class labels for given input data."""
        A2 = self.forward_prop(X)
        return np.argmax(A2, axis=0)
