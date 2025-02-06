import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.utils.data_loader import load_test_data
from src.models.neural_network import NeuralNetwork


def load_weights(filename="model_weights.npz"):
    """
    Loads the trained model weights from a file.

    Args:
        filename (str): Name of the file to load the weights from.

    Returns:
        tuple: Weights and biases for the network.
    """
    try:
        data = np.load(filename)
        return data["W1"], data["b1"], data["W2"], data["b2"]
    except FileNotFoundError:
        print(f"File not found. Train the model first.")


def evaluate(sample_size=1000):
    """
    Evaluates the trained model on the test dataset.

    Args:
        sample_size (int): Number of samples to evaluate on.

    Returns:
        float: Accuracy of the model on the test dataset
    """
    print(f"Evaluating with {sample_size} test samples")
    X_test, y_test = load_test_data(sample_size)

    W1, b1, W2, b2 = load_weights()

    nn = NeuralNetwork(W1, b1, W2, b2)
    predictions = nn.predict(X_test)

    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy
