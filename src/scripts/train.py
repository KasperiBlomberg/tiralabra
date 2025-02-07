import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import numpy as np
from src.utils.data_loader import load_train_data
from src.utils.training import init_params, update_params, backward_prop
from src.models.neural_network import NeuralNetwork


def train(
    alpha=0.1,
    iterations=1000,
    sample_size=1000,
    filename="model_weights.npz",
    batch_size=128,
):
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
    W1, b1, W2, b2, W3, b3 = init_params()
    num_batches = X_train.shape[1] // batch_size

    for i in range(iterations):
        X_train_batches = np.array_split(X_train, num_batches, axis=1)
        Y_train_batches = np.array_split(Y_train, num_batches, axis=0)
        for batch in range(num_batches):
            X_train_batch = X_train_batches[batch]
            Y_train_batch = Y_train_batches[batch]
            nn = NeuralNetwork(W1, b1, W2, b2, W3, b3)
            A3 = nn.forward_prop(X_train_batch)
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(
                nn.Z1,
                nn.A1,
                nn.Z2,
                nn.A2,
                nn.Z3,
                A3,
                W1,
                W2,
                W3,
                X_train_batch,
                Y_train_batch,
            )
            W1, b1, W2, b2, W3, b3 = update_params(
                W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha
            )

        if i % 10 == 0 or i == iterations - 1:
            predictions = nn.predict(X_train)
            accuracy = np.mean(predictions == Y_train)
            print(f"Iteration {i} | Training Accuracy: {accuracy:.4f}")

    save_params(W1, b1, W2, b2, W3, b3, filename)


def save_params(W1, b1, W2, b2, W3, b3, filename="model_weights.npz"):
    """
    Saves the trained model weights to a file.

    Args:
        W1 (numpy.ndarray): Weights of the first layer.
        b1 (numpy.ndarray): Biases of the first layer.
        W2 (numpy.ndarray): Weights of the second layer.
        b2 (numpy.ndarray): Biases of the second layer.
        W3 (numpy.ndarray): Weights of the third layer.
        b3 (numpy.ndarray): Biases of the third layer.
        filename (str): Name of the file to save the weights to.

    Note:
        The function doesn't return anything but it saves the weights to a file.
    """
    np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
