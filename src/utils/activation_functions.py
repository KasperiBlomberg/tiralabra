import numpy as np


def ReLU(Z):
    """ReLU function. Returns a matrix of the same shape as Z.
    Every negative value is replaced by 0 and positive values stay the same."""
    return np.maximum(Z, 0)


def ReLU_deriv(Z):
    """Derivative of the ReLU function.
    Returns a boolean matrix of the same shape as Z.
    Booleans behave like 0s and 1s in calculations."""
    return Z > 0


def softmax(Z):
    """
    Computes the softmax of the columns of Z.
    Basically makes every column of Z into a probability distribution.
    Uses a numerical stability shift to avoid overflow.
    """
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shifted)
    return expZ / np.sum(expZ, axis=0, keepdims=True)
