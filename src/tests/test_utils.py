import unittest
import numpy as np
from src.utils.data_loader import one_hot, load_train_data, load_test_data
from src.utils.training import init_params, backward_prop, update_params


class TestUtils(unittest.TestCase):
    def test_one_hot(self):
        Y = np.array([0, 1, 2])
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(one_hot(Y), expected)

    def test_load_train_data(self):
        X_train, y_train = load_train_data(1000)
        self.assertEqual(X_train.shape, (784, 1000))
        self.assertEqual(y_train.shape, (1000,))

    def test_load_test_data(self):
        X_test, y_test = load_test_data(1000)
        self.assertEqual(X_test.shape, (784, 1000))
        self.assertEqual(y_test.shape, (1000,))

    def test_init_params(self):
        W1, b1, W2, b2, W3, b3 = init_params()
        self.assertEqual(W1.shape, (10, 784))
        self.assertEqual(b1.shape, (10, 1))
        self.assertEqual(W2.shape, (10, 10))
        self.assertEqual(b2.shape, (10, 1))
        self.assertEqual(W3.shape, (10, 10))
        self.assertEqual(b3.shape, (10, 1))

    def test_backward_prop(self):
        pass

    def test_update_params(self):
        pass
