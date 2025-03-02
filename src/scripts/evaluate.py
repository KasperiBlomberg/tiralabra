import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from src.utils.data_loader import load_test_data, load_weights
from src.models.neural_network import NeuralNetwork


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

    nn = NeuralNetwork(*load_weights(filename))
    predictions = nn.predict(X_test)

    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy


def view_predictions(sample_size=10, filename="model_weights.npz"):
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

    nn = NeuralNetwork(*load_weights(filename))
    predictions = nn.predict(X_test)

    for i in range(sample_size):
        prediction = labels[predictions[i]]
        actual = labels[y_test[i]]
        if prediction == actual:
            color = "green"
            print(f"Correct! Prediction: {prediction}, Actual: {actual}")
        else:
            color = "red"
            print(f"Wrong! Prediction: {prediction}, Actual: {actual}")

        plt.imshow(X_test[:, i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.title(f"Prediction: {prediction}\nActual: {actual}", color=color, fontsize=14)
        plt.show()
