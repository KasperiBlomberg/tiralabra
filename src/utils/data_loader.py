import os
from src.utils import mnist_reader
import numpy as np

def load_train_data(sample_size):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data/fashion')

    X_train, y_train = mnist_reader.load_mnist(DATA_DIR, kind='train')

    X_train, y_train = X_train[:sample_size], y_train[:sample_size]
    X_train, y_train = X_train.T, y_train.T
    X_train = X_train / 255

    return X_train, y_train

def load_test_data(sample_size):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    DATA_DIR = os.path.join(BASE_DIR, 'data/fashion')

    X_test, y_test = mnist_reader.load_mnist(DATA_DIR, kind='t10k')

    X_test, y_test = X_test[:sample_size], y_test[:sample_size]
    X_test, y_test = X_test.T, y_test.T
    X_test = X_test / 255

    return X_test, y_test

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

if __name__ == "__main__":
    pass