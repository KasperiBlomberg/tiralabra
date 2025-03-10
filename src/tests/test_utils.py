import unittest
import numpy as np
from src.utils.data_loader import one_hot, load_train_data, load_test_data
from src.utils.activation_functions import ReLU, ReLU_deriv, softmax


class TestUtils(unittest.TestCase):
    def test_one_hot(self):
        Y = np.array([0, 1, 2])
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(one_hot(Y, 3), expected)

    def test_load_train_data(self):
        X_train, y_train = load_train_data(1000)
        self.assertEqual(X_train.shape, (784, 1000))
        self.assertEqual(y_train.shape, (1000,))

    def test_load_test_data(self):
        X_test, y_test = load_test_data(1000)
        self.assertEqual(X_test.shape, (784, 1000))
        self.assertEqual(y_test.shape, (1000,))

    def test_relu(self):
        Z = np.array([[-1, 0, 5], [3, -2, 0]])
        expected = np.array([[0, 0, 5], [3, 0, 0]])
        output = ReLU(Z)
        np.testing.assert_array_equal(output, expected)

    def test_relu_deriv(self):
        Z = np.array([[-1, 0, 5], [3, -2, 0]])
        expected = np.array([[False, False, True], [True, False, False]])
        output = ReLU_deriv(Z)
        np.testing.assert_array_equal(output, expected)

    def test_softmax(self):
        """
        Test the column-wise softmax on a small 2D example.
        2 x 3 array => 2 samples, 3 classes.
        """
        Z = np.array([[1, 2, 3], [4, 5, 6]])

        expected = np.array(
            [[0.04742587, 0.04742587, 0.04742587], [0.95257413, 0.95257413, 0.95257413]]
        )

        output = softmax(Z)
        np.testing.assert_allclose(output, expected, rtol=1e-6)

        # Check that each column sums to 1
        column_sums = np.sum(output, axis=0)
        np.testing.assert_allclose(column_sums, np.ones(output.shape[1]), rtol=1e-6)
