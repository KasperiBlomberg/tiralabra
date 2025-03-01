import unittest
import numpy as np

from src.models.neural_network import NeuralNetwork


class TestNeuralnetwork(unittest.TestCase):
    def setUp(self):
        """
        Set up a neural network with 5 neurons in each layer and 10 samples and 20 features and 3 classes.
        """
        np.random.seed(0)

        W1 = np.random.randn(5, 20)
        b1 = np.random.randn(5, 1)
        W2 = np.random.randn(5, 5)
        b2 = np.random.randn(5, 1)
        W3 = np.random.randn(3, 5)
        b3 = np.random.randn(3, 1)

        self.nn = NeuralNetwork(W1, b1, W2, b2, W3, b3)
        self.X = np.random.randn(20, 10)

    def test_forward_prop(self):
        """
        Test the forward propagation method.
        """
        A3 = self.nn.forward_prop(self.X)
        self.assertEqual(A3.shape, (3, 10))

        column_sums = np.sum(A3, axis=0)
        np.testing.assert_allclose(column_sums, np.ones(A3.shape[1]), rtol=1e-6)

    def test_forward_prop_all_zeros(self):
        """
        Test the forward propagation method with zeros.
        """
        X = np.zeros((20, 10))
        A3 = self.nn.forward_prop(X)
        self.assertEqual(A3.shape, (3, 10))

        column_sums = np.sum(A3, axis=0)
        np.testing.assert_allclose(column_sums, np.ones(A3.shape[1]), rtol=1e-6)
