import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import numpy as np
from src.utils.data_loader import load_train_data
from src.models.neural_network import NeuralNetwork


def train(
    alpha=0.1,
    iterations=100,
    sample_size=30000,
    filename="testi.npz",
    batch_size=128,
):
    """
    Trains the neural network on the dataset.
    Uses mini-batches with shuffling.

    Args:
        alpha (float): Learning rate for the model.
        iterations (int): Number of iterations to train the model.
        sample_size (int): Number of samples to train on.

    Note:
        The function doesn't return anything but it saves the trained model weights to a file.
    """

    X_train, Y_train = load_train_data(sample_size)
    num_batches = X_train.shape[1] // batch_size
    nn = NeuralNetwork(randomize=True)

    for i in range(iterations):
        permutation = np.random.permutation(X_train.shape[1])
        X_train = X_train[:, permutation]
        Y_train = Y_train[permutation]
        X_train_batches = np.array_split(X_train, num_batches, axis=1)
        Y_train_batches = np.array_split(Y_train, num_batches, axis=0)

        for batch in range(num_batches):
            X_train_batch = X_train_batches[batch]
            Y_train_batch = Y_train_batches[batch]

            nn.forward_prop(X_train_batch)

            nn.backward_prop(
                X_train_batch,
                Y_train_batch,
            )

            nn.update_params(alpha)

        if i % 10 == 0 or i == iterations - 1:
            predictions = nn.predict(X_train)
            accuracy = np.mean(predictions == Y_train)
            print(f"Iteration {i} | Training Accuracy: {accuracy:.4f}")

    nn.save_params(filename)
