import numpy as np
from src.utils.activation_functions import ReLU, softmax


class NeuralNetwork:
    """
    A simple two-layer neural network for classification.

    Attributes:
        W1 (numpy.ndarray): Weights of the first layer.
        b1 (numpy.ndarray): Biases of the first layer.
        W2 (numpy.ndarray): Weights of the second layer.
        b2 (numpy.ndarray): Biases of the second layer.
    """

    def __init__(self, W1, b1, W2, b2):
        """
        Initializes the neural network with given weights and biases.
        """
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2

    def forward_prop(self, X):
        """
        Performs forward propagation to compute the output of the network.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted class probabilities.
        """
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = ReLU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def predict(self, X):
        """
        Predicts the class of input data.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        A2 = self.forward_prop(X)
        return np.argmax(A2, axis=0)
