import numpy as np


def ReLU(Z):
    """
    ReLU function.
    Every negative value is replaced by 0 and positive values stay the same.

    Args:
        Z (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Output of the ReLU function.
    """
    return np.maximum(Z, 0)


def ReLU_deriv(Z):
    """
    Computes the derivative of the ReLU function.

    Args:
        Z (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: A boolean matrix where positive values are True and non-positive values are False.
        This matrix behaves like 1s and 0s in calculations.
    """
    return Z > 0


def softmax(Z):
    """
    Computes the softmax of the columns of Z.
    Basically makes every column of Z into a probability distribution.
    Uses a numerical stability shift to avoid overflow.

    Args:
        Z (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Output of the softmax function.
    """
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shifted)
    return expZ / np.sum(expZ, axis=0, keepdims=True)
