import unittest
import numpy as np
from src.utils.data_loader import one_hot, load_train_data, load_test_data

class TestUtils(unittest.TestCase):

    def test_one_hot(self):
        Y = np.array([0, 1, 2])
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(one_hot(Y), expected)

    def test_load_train_data(self):
        X_train, y_train = load_train_data(1000)
        self.assertEqual(X_train.shape, (784, 1000))
        self.assertEqual(y_train.shape, (1000,))

    def test_load_test_data(self):
        X_test, y_test = load_test_data(1000)
        self.assertEqual(X_test.shape, (784, 1000))
        self.assertEqual(y_test.shape, (1000,))