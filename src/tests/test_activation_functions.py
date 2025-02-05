import unittest
import numpy as np

from src.utils.activation_functions import ReLU, ReLU_deriv, softmax


class TestActivationFunctions(unittest.TestCase):
    def test_relu(self):
        # Create a sample input
        Z = np.array([[-1, 0, 5],
                      [3, -2, 0]])
        # Expected output: negative values become 0, positives stay the same
        expected = np.array([[0, 0, 5],
                             [3, 0, 0]])
        output = ReLU(Z)
        np.testing.assert_array_equal(output, expected)

    def test_relu_deriv(self):
        # Create a sample input
        Z = np.array([[-1, 0, 5],
                      [3, -2, 0]])
        # Expected output: if Z > 0 => True, else False
        expected = np.array([[False, False, True],
                             [True,  False, False]])
        output = ReLU_deriv(Z)
        np.testing.assert_array_equal(output, expected)

    def test_softmax_2d(self):
        """
        Test the column-wise softmax on a small 2D example.
        2 x 3 array => 2 samples, 3 classes.
        """
        Z = np.array([[1, 2, 3],
                      [4, 5, 6]])

        # Manual calculation column-by-column:
        # For each column, subtract max, exponentiate, and normalize.
        # Expected:
        #   Column 0 => [1,4], shift => [-3,0], => [exp(-3), exp(0)] => [0.0498, 1], 
        #     sum=1.0498 => normalized => [0.0474, 0.9526]
        #   Column 1 => [2,5], shift => [-3,0], => same pattern => [0.0474, 0.9526]
        #   Column 2 => [3,6], shift => [-3,0], => same pattern => [0.0474, 0.9526]
        #
        # So we expect each column to be ~ [0.0474, 0.9526].
        expected = np.array([
            [0.04742587, 0.04742587, 0.04742587],
            [0.95257413, 0.95257413, 0.95257413]])

        output = softmax(Z)
        np.testing.assert_allclose(output, expected, rtol=1e-6)

        # Check that each column sums to 1 TODO
