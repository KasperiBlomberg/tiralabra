import numpy as np
from src.utils.activation_functions import ReLU, softmax, ReLU_deriv
from src.utils.data_loader import one_hot


class NeuralNetwork:
    """
    Three-layer neural network for classification.

    Attributes:
        W1 (numpy.ndarray): Weights of the first layer.
        b1 (numpy.ndarray): Biases of the first layer.
        W2 (numpy.ndarray): Weights of the second layer.
        b2 (numpy.ndarray): Biases of the second layer.
        W3 (numpy.ndarray): Weights of the third layer.
        b3 (numpy.ndarray): Biases of the third layer.
        randomize (bool): Whether to use a random seed for initialization.
    """

    def __init__(self, W1=None, b1=None, W2=None, b2=None, W3=None, b3=None):
        """
        Initializes the neural network with given weights and biases.
        """
        if any(param is None for param in (W1, b1, W2, b2, W3, b3)):
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self._init_params()
        else:
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = (
                W1,
                b1,
                W2,
                b2,
                W3,
                b3,
            )

    @staticmethod
    def _init_params():
        """
        Initialize weights and biases fo training.
        """
        return (
            np.random.rand(256, 784) - 0.5,
            np.random.rand(256, 1) - 0.5,
            np.random.rand(128, 256) - 0.5,
            np.random.rand(128, 1) - 0.5,
            np.random.rand(10, 128) - 0.5,
            np.random.rand(10, 1) - 0.5,
        )

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
        self.A2 = ReLU(self.Z2)

        self.Z3 = self.W3.dot(self.A2) + self.b3
        self.A3 = softmax(self.Z3)

        return self.A3

    def backward_prop(self, X, Y):
        """
        Backward propagation to get the gradients of the loss with respect to the weights and biases.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.

        Returns:
            tuple: Gradients of the loss with respect to the weights and biases.
        """
        one_hot_Y = one_hot(Y)
        m = Y.size
        dZ3 = self.A3 - one_hot_Y
        self.dW3 = 1 / m * dZ3.dot(self.A2.T)
        self.db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
        dZ2 = self.W3.T.dot(dZ3) * ReLU_deriv(self.Z2)
        self.dW2 = 1 / m * dZ2.dot(self.A1.T)
        self.db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = self.W2.T.dot(dZ2) * ReLU_deriv(self.Z1)
        self.dW1 = 1 / m * dZ1.dot(X.T)
        self.db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    def update_params(self, alpha, lambda_reg=0.001):
        """
        Update the weights and biases of the network.

        Args:

            alpha (float): Learning rate.
            lambda_reg (float): L2 regularization parameter.

        Returns:
            tuple: Updated weights and biases for the network.
        """
        self.W1 = self.W1 - alpha * (self.dW1 + lambda_reg * self.W1)
        self.b1 = self.b1 - alpha * self.db1
        self.W2 = self.W2 - alpha * (self.dW2 + lambda_reg * self.W2)
        self.b2 = self.b2 - alpha * self.db2
        self.W3 = self.W3 - alpha * (self.dW3 + lambda_reg * self.W3)
        self.b3 = self.b3 - alpha * self.db3

    def predict(self, X):
        """
        Predicts the class of input data.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        self.forward_prop(X)
        return np.argmax(self.A3, axis=0)

    def save_params(self, filename="model_weights.npz"):
        """
        Saves model weights to a file.
        """
        np.savez(
            filename,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
        )
