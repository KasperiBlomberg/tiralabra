import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
import numpy as np

from src.utils.data_loader import load_train_data, one_hot, load_weights, load_test_data
from src.models.neural_network import NeuralNetwork


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X_train, self.Y_train = load_train_data(10)
        self.nn = NeuralNetwork()
        self.iterations = 100
        self.alpha = 0.01
        self.batch_size = 5
        self.loss_history = []
        self.accuracy_history = []

    def test_train(self):
        """
        Method train from scripts copied but modified a bit to be able to test it.
        """

        X_train, Y_train = self.X_train, self.Y_train
        num_samples = X_train.shape[1]
        num_batches = num_samples // self.batch_size
        nn = self.nn

        for i in range(1, self.iterations + 1):
            permutation = np.random.permutation(num_samples)

            for j in range(num_batches):
                batch_indices = permutation[j * self.batch_size:(j + 1) * self.batch_size]
                X_train_batch = X_train[:, batch_indices]
                Y_train_batch = Y_train[batch_indices]
                
                nn.forward_prop(X_train_batch)

                nn.backward_prop(X_train_batch, Y_train_batch)

                W1_old, b1_old, W2_old, b2_old, W3_old, b3_old = (
                    nn.W1,
                    nn.b1,
                    nn.W2,
                    nn.b2,
                    nn.W3,
                    nn.b3,
                )

                nn.update_params(self.alpha)

                # Check that gradients are not zero
                self.assertTrue(
                    np.any(nn.dW1 != 0) or np.any(nn.db1 != 0),
                    "Gradients for Layer 1 should not be zero",
                )
                self.assertTrue(
                    np.any(nn.dW2 != 0) or np.any(nn.db2 != 0),
                    "Gradients for Layer 2 should not be zero",
                )
                self.assertTrue(
                    np.any(nn.dW3 != 0) or np.any(nn.db3 != 0),
                    "Gradients for Layer 3 should not be zero",
                )

                # Check that weights and biases are updated after each iteration
                self.assertFalse(
                    np.all(nn.W1 == W1_old), "W1 should update each iteration"
                )
                self.assertFalse(
                    np.all(nn.b1 == b1_old), "b1 should update each iteration"
                )
                self.assertFalse(
                    np.all(nn.W2 == W2_old), "W2 should update each iteration"
                )
                self.assertFalse(
                    np.all(nn.b2 == b2_old), "b2 should update each iteration"
                )
                self.assertFalse(
                    np.all(nn.W3 == W3_old), "W3 should update each iteration"
                )
                self.assertFalse(
                    np.all(nn.b3 == b3_old), "b3 should update each iteration"
                )

            if i % 10 == 0 or i == 1:
                predictions = nn.predict(self.X_train)
                accuracy = np.mean(predictions == self.Y_train)
                loss = (
                    -np.sum(one_hot(self.Y_train) * np.log(nn.A3)) / self.Y_train.size
                )

                self.loss_history.append(loss)
                self.accuracy_history.append(accuracy)

        # Check that loss decreases over iterations
        self.assertTrue(
            self.loss_history[-1] < self.loss_history[0],
            "Loss should decrease during training",
        )

        # Check that network overfits (training accuracy is 1)
        self.assertTrue(
            self.accuracy_history[-1] == 1,
            "Training accuracy should be high, indicating overfitting",
        )

    def test_order_of_samples(self):
        """
        Test if the order of samples in the dataset affects the output.
        """

        X_test, _ = load_test_data(5)
        nn = NeuralNetwork(load_weights())

        batch_output = nn.predict(X_test)

        individual_outputs = []
        for i in range(X_test.shape[1]):
            output = nn.predict(X_test[:, i].reshape(-1, 1))[0]
            individual_outputs.append(output)

        assert np.array_equal(
            batch_output, individual_outputs
        ), "Batch and individual outputs should match"
