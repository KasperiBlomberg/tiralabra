import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import numpy as np
from src.utils.data_loader import load_train_data
from src.utils.training import init_params, update_params, backward_prop
from src.models.neural_network import NeuralNetwork


def train(alpha=0.1, iterations=1000, sample_size=1000, filename="model_weights.npz"):
    """
    Trains the neural network on the dataset.

    Args:
        alpha (float): Learning rate for the model.
        iterations (int): Number of iterations to train the model.
        sample_size (int): Number of samples to train on.

    Note:
        The function doesn't return anything but it saves the trained model weights to a file.
    """
    X_train, Y_train = load_train_data(sample_size)
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        nn = NeuralNetwork(W1, b1, W2, b2)
        A2 = nn.forward_prop(X_train)
        dW1, db1, dW2, db2 = backward_prop(
            nn.Z1, nn.A1, nn.Z2, A2, W1, W2, X_train, Y_train
        )
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = nn.predict(X_train)
            accuracy = np.mean(predictions == Y_train)
            print(f"Iteration {i} | Training Accuracy: {accuracy:.4f}")

    save_params(W1, b1, W2, b2, filename)


def save_params(W1, b1, W2, b2, filename="model_weights.npz"):
    """
    Saves the trained model weights to a file.

    Args:
        W1 (numpy.ndarray): Weights of the first layer.
        b1 (numpy.ndarray): Biases of the first layer.
        W2 (numpy.ndarray): Weights of the second layer.
        b2 (numpy.ndarray): Biases of the second layer.
        filename (str): Name of the file to save the weights to.

    Note:
        The function doesn't return anything but it saves the weights to a file.
    """
    np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)
