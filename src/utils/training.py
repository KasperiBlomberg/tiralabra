import numpy as np
from src.utils.activation_functions import ReLU_deriv
from src.utils.data_loader import one_hot


def init_params():
    """
    Initializes the weights and biases for the network.

    Returns:
        tuple: Weights and biases for the network.
    """
    W1 = np.random.rand(256, 784) - 0.5
    b1 = np.random.rand(256, 1) - 0.5
    W2 = np.random.rand(128, 256) - 0.5
    b2 = np.random.rand(128, 1) - 0.5
    W3 = np.random.rand(10, 128) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3


def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    """
    Backward propagation to get the gradients of the loss with respect to the weights and biases.

    Args:
        Z1 (numpy.ndarray): Output of the first layer before activation.
        A1 (numpy.ndarray): Output of the first layer after activation.
        Z2 (numpy.ndarray): Output of the second layer before activation.
        A2 (numpy.ndarray): Output of the second layer after activation.
        Z3 (numpy.ndarray): Output of the third layer before activation.
        A3 (numpy.ndarray): Output of the third layer after activation.
        W1 (numpy.ndarray): Weights of the first layer.
        W2 (numpy.ndarray): Weights of the second layer.
        W3 (numpy.ndarray): Weights of the third layer.
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): True labels.

    Returns:
        tuple: Gradients of the loss with respect to the weights and biases.
    """
    one_hot_Y = one_hot(Y)
    m = Y.size
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(
        Z1
    )  # for sigmoid dZ1 = W2.T.dot(dZ2) * sigmoid_derivative(A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    """
    Update the weights and biases of the network.

    Args:
        W1 (numpy.ndarray): First layer weights.
        b1 (numpy.ndarray): First layer biases.
        W2 (numpy.ndarray): Second layer weights.
        b2 (numpy.ndarray): Second layer biases.
        W3 (numpy.ndarray): Third layer weights.
        b3 (numpy.ndarray): Third layer biases.
        dW1 (numpy.ndarray): Gradient of first layer weights.
        db1 (numpy.ndarray): Gradient of first layer biases.
        dW2 (numpy.ndarray): Gradient of second layer weights.
        db2 (numpy.ndarray): Gradient of second layer biases.
        dW3 (numpy.ndarray): Gradient of third layer weights.
        db3 (numpy.ndarray): Gradient of third layer biases.
        alpha (float): Learning rate

    Returns:
        tuple: Updated weights and biases for the network.
    """
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3
