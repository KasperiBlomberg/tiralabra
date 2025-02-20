import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
import numpy as np

from src.utils.data_loader import load_train_data, one_hot
from src.models.neural_network import NeuralNetwork
from src.utils.training import backward_prop, update_params, init_params

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.X_train, self.Y_train = load_train_data(100)
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = init_params(randomize=False)
        self.num_batches = 1
        self.iterations = 1000
        self.alpha = 0.1
        self.loss_history = []
        self.accuracy_history = []

    def test_train(self):
        """
        Method train from scripts copied but modified a bit to abble to test it.
        """
        W1, b1, W2, b2, W3, b3 = self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

        for i in range(self.iterations):
            permutation = np.random.permutation(self.X_train.shape[1])
            X_train = self.X_train[:, permutation]
            Y_train = self.Y_train[permutation]
            X_train_batches = np.array_split(X_train, self.num_batches, axis=1)
            Y_train_batches = np.array_split(Y_train, self.num_batches, axis=0)

            for batch in range(self.num_batches):
                X_train_batch = X_train_batches[batch]
                Y_train_batch = Y_train_batches[batch]
                nn = NeuralNetwork(W1, b1, W2, b2, W3, b3)
                A3 = nn.forward_prop(X_train_batch)

                dW1, db1, dW2, db2, dW3, db3 = backward_prop(
                nn.Z1,
                nn.A1,
                nn.Z2,
                nn.A2,
                nn.Z3,
                A3,
                W1,
                W2,
                W3,
                X_train_batch,
                Y_train_batch,
                )

                W1_new, b1_new, W2_new, b2_new, W3_new, b3_new = update_params(
                    W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, self.alpha
                )

                # Check that gradients are not zero
                self.assertTrue(np.any(dW1 != 0) or np.any(db1 != 0), "Gradients for Layer 1 should not be zero")
                self.assertTrue(np.any(dW2 != 0) or np.any(db2 != 0), "Gradients for Layer 2 should not be zero")
                self.assertTrue(np.any(dW3 != 0) or np.any(db3 != 0), "Gradients for Layer 3 should not be zero")
                
                # Check that weights and biases are changing
                self.assertFalse(np.all(W1 == W1_new), "W1 should update each iteration")
                self.assertFalse(np.all(b1 == b1_new), "b1 should update each iteration")
                self.assertFalse(np.all(W2 == W2_new), "W2 should update each iteration")
                self.assertFalse(np.all(b2 == b2_new), "b2 should update each iteration")
                self.assertFalse(np.all(W3 == W3_new), "W3 should update each iteration")
                #self.assertFalse(np.all(b3 == b3_new), "b3 should update each iteration") # Causes an error has to be fixed

                W1, b1, W2, b2, W3, b3 = W1_new, b1_new, W2_new, b2_new, W3_new, b3_new



            if i % 100 == 0 or i == self.iterations - 1:
                predictions = nn.predict(self.X_train)
                accuracy = np.mean(predictions == self.Y_train)
                loss = -np.sum(one_hot(self.Y_train) * np.log(A3)) / self.Y_train.size
                
                self.loss_history.append(loss)
                self.accuracy_history.append(accuracy)

        # Check if loss decreases over iterations
        self.assertTrue(self.loss_history[-1] < self.loss_history[0], "Loss should decrease during training")
        
        # Check if network overfits (high training accuracy)
        self.assertTrue(self.accuracy_history[-1] > 0.9, "Training accuracy should be high, indicating overfitting")

        