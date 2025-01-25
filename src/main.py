import mnist_reader
import numpy as np


#X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
#X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

def preprocess_data(X_train, X_test):
    """Preprocess the data by normalizing the pixel values in the range [0, 1]."""
    return X_train / 255, X_test / 255

#x_train, x_test = preprocess_data(X_train, X_test)

#TODO: test
def one_hot_encode(y):
    """Encode labels into one-hot encoding. 
    This function takes an array of integer labels and converts them into a one-hot encoded format, 
    where each label is represented by a binary vector of length 10 (for a 10-class classification problem).
    
    For example, if y = 3, the one-hot encoding is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
    
    Args:
        y (array-like): Array of integer labels to encode. Each label should be an integer in the range [0, 9].

    Returns:
        numpy.ndarray: A 2D array where each row corresponds to the one-hot encoded vector of a label in `y`."""
    classes = 10
    return np.eye(classes)[y]

def decode_one_hot(y):
    """Decode one-hot encoded label into integer label."""
    return np.argmax(y)

#y_train = one_hot_encode(y_train)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(x):
    """Compute the derivative of the sigmoid function."""
    return x * (1 - x)

def softmax(z):
    """TODO: add docstring"""
    return np.exp(z) / sum(np.exp(z))

def softmax_derivative(a, y):
    """Compute the derivative of the softmax function."""
    return a - y

def create_batches(X, y, batch_size):
    """Create batches of data samples and corresponding labels."""
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]

def feedforward(X, weights, biases, activation, output_activation):
    """Perform feedforward operation in a neural network."""
    a = X
    zs = []
    activations = [a]
    for i in range(len(weights)):
        z = np.dot(a, weights[i]) + biases[i]
        zs.append(z)
        if i == len(weights) - 1: # Output layer
            a = output_activation(z)
        else:
            a = activation(z)
        activations.append(a)
    return zs, activations

def error(true, predicted):
    """Compute the mean squared error between the true and predicted values."""
    return np.mean((true - predicted) ** 2)

def backpropagation(X, y, weights, activations, zs):
    """Perform backpropagation to compute gradients for weights and biases."""
    dzs = []
    dWs = []
    dbs = []
    m = X.shape[0]
    pass

def update_parameters(weights, biases, dWs, dbs, learning_rate):
    """Update the weights and biases of the neural network."""
    for l in range(len(weights)):
        weights[l] -= learning_rate * dWs[l]
        biases[l] -= learning_rate * dbs[l]

    return weights, biases

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, y, weights, biases, learning_rate, iterations):
    """Perform gradient descent to train the neural network."""
    for i in range(iterations):
        zs, activations = feedforward(X, weights, biases, sigmoid, softmax)
        dWs, dbs = backpropagation(X, y, weights, activations, zs)
        weights, biases = update_parameters(weights, biases, dWs, dbs, learning_rate)
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(activations[-1])
            print(get_accuracy(predictions, y))

    return weights, biases


# TODO: Implement parameters to choose the number of hidden layers and units in each layer.
def initialize_parameters():
    """Initialize the weights and biases of the neural network."""
    weights = [
        np.random.rand(784, 10) - 0.5, # -0.5 to center the values around 0 which helps in convergence
        np.random.rand(10, 10) - 0.5
    ]

    biases = [
        np.random.randn(1, 10) - 0.5,
        np.random.randn(1, 10) - 0.5
    ]

    return weights, biases


if __name__ == "__main__":
    # test the feedforward function
    # x = np.array([0.1, 0.5])
    # weights = [
    #     np.array([[0.1, 0.3], [0.2, 0.4]]).T,
    #     np.array([[0.5, 0.6], [0.7, 0.8]]).T
    # ]
    # biases = [
    #     np.array([0.25, 0.25]),
    #     np.array([0.35, 0.35])
    # ]
    # y_true = np.array([0.05, 0.95])

    # feedforward_output = feedforward(x, weights, biases, sigmoid, softmax)

    # print(feedforward_output) # Expected output: [0.73492 0.77955]
    #print(error(y_true, feedforward_output)) # Expected output: 0.24908

    # updated_weights, updated_biases = backpropagation(x, y_true, weights, biases)

    # print("Updated weights:", updated_weights)
    # print("Updated biases:", updated_biases)


    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    #x_train, x_test = preprocess_data(X_train, X_test)
    y_train = y_train[:1000]
    x_train = X_train[:1000]
    y_train = one_hot_encode(y_train)

    
    #print(x_train.shape, y_train.shape)

    # test the feedforward function
    weights, biases = initialize_parameters()
    #print(feedforward(x_train[0], weights, biases, sigmoid, softmax))

    #weights, biases = gradient_descent(x_train, y_train, weights, biases, 0.1, 500)