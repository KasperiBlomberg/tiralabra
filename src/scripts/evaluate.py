import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
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
        return data["W1"], data["b1"], data["W2"], data["b2"], data["W3"], data["b3"]
    except FileNotFoundError:
        print(f"File not found. Train the model first.")


def evaluate(sample_size=1000, filename="model_weights.npz"):
    """
    Evaluates the trained model on the test dataset.

    Args:
        sample_size (int): Number of samples to evaluate on.

    Returns:
        float: Accuracy of the model on the test dataset
    """
    print(f"Evaluating with {sample_size} test samples")
    X_test, y_test = load_test_data(sample_size)

    W1, b1, W2, b2, W3, b3 = load_weights(filename)

    nn = NeuralNetwork(W1, b1, W2, b2, W3, b3)
    predictions = nn.predict(X_test)

    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy


def view_predictions(sample_size=10):
    """
    View the model predictions on a sample of the test dataset.

    Args:
        sample_size (int): Number of samples to view.
    """
    labels = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    X_test, y_test = load_test_data()
    indices = np.random.choice(X_test.shape[1], sample_size, replace=False)
    X_test = X_test[:, indices]
    y_test = y_test[indices]

    W1, b1, W2, b2, W3, b3 = load_weights()

    nn = NeuralNetwork(W1, b1, W2, b2, W3, b3)
    predictions = nn.predict(X_test)

    for i in range(sample_size):
        prediction = labels[predictions[i]]
        actual = labels[y_test[i]]
        if prediction == actual:
            print(f"Correct! Prediction: {prediction}, Actual: {actual}")
        else:
            print(f"Wrong! Prediction: {prediction}, Actual: {actual}")

        plt.imshow(X_test[:, i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.show()
