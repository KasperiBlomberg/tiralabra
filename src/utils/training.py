import numpy as np
from src.utils.activation_functions import ReLU_deriv
from src.utils.data_loader import one_hot


def init_params():
    """
    Initializes the weights and biases for the network.

    Returns:
        tuple: Weights and biases for the network.
    """
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    """
    Backward propagation to get the gradients of the loss with respect to the weights and biases.

    Args:
        Z1 (numpy.ndarray): Output of the first layer before activation.
        A1 (numpy.ndarray): Output of the first layer after activation.
        Z2 (numpy.ndarray): Output of the second layer before activation.
        A2 (numpy.ndarray): Output of the second layer after activation.
        W1 (numpy.ndarray): Weights of the first layer.
        W2 (numpy.ndarray): Weights of the second layer.
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): True labels.

    Returns:
        tuple: Gradients of the loss with respect to the weights and biases.
    """
    one_hot_Y = one_hot(Y)
    m = Y.size
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(
        Z1
    )  # for sigmoid dZ1 = W2.T.dot(dZ2) * sigmoid_derivative(A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Update the weights and biases of the network.

    Args:
        W1 (numpy.ndarray): First layer weights.
        b1 (numpy.ndarray): First layer biases.
        W2 (numpy.ndarray): Second layer weights.
        b2 (numpy.ndarray): Second layer biases.
        dW1 (numpy.ndarray): Gradient of first layer weights.
        db1 (numpy.ndarray): Gradient of first layer biases.
        dW2 (numpy.ndarray): Gradient of second layer weights.
        db2 (numpy.ndarray): Gradient of second layer biases.
        alpha (float): Learning rate

    Returns:
        tuple: Updated weights and biases for the network.
    """
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2
