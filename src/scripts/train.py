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
    filename="model_weights.npz",
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
    num_samples = X_train.shape[1]
    num_batches = num_samples // batch_size
    nn = NeuralNetwork()

    for i in range(1, iterations + 1):
        permutation = np.random.permutation(num_samples)

        for j in range(num_batches):
            batch_indices = permutation[j * batch_size : (j + 1) * batch_size]
            X_train_batch = X_train[:, batch_indices]
            Y_train_batch = Y_train[batch_indices]

            nn.forward_prop(X_train_batch)

            nn.backward_prop(
                X_train_batch,
                Y_train_batch,
            )

            nn.update_params(alpha)

        if i % 10 == 0 or i == 1:
            predictions = nn.predict(X_train)
            accuracy = np.mean(predictions == Y_train)
            print(f"Iteration {i} | Training Accuracy: {accuracy:.4f}")

    nn.save_params(filename)
