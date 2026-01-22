# Implementation Document
## General Structure of the Program
The program recognizes clothes and accessories from images into different categories. The core of the program is a multilayer perceptron neural network. The program is implemented in Python, and the NumPy library is used for matrix calculations. The neural network consists of an input layer of 784 neurons, two hidden layers (256 and 128 neurons), and an output layer with 10 neurons, i.e., one neuron for each class. The network uses ReLU as the activation function and softmax in the output layer.

The network uses L2 regularization in training for updating parameters so that weights do not grow too large. Mini-batches are also used in training, and the order of batches is shuffled in every iteration.

At best, the program has achieved approx. 94% accuracy on training data and approx. 89% accuracy on test data. According to Zalando, humans recognize images correctly 83.5% of the time, and a relatively similar MLP (layer sizes 256-128-100) has recognized 88% of images, so the results are quite good.

The program consists of the following components:
- `neural_network.py` The main component, which contains the structure of the neural network and methods for the neural network's operation.
- `scripts` Contains scripts for training and evaluating the neural network.
- `activation_functions.py` Contains the activation functions needed by the neural network.
- `data_loader.py` and mnist_reader.py The mnistreader file is directly copied from Zalando's repo and is used to load images from the file. The Data_loader component contains functions for data preprocessing, for example, for normalization and label encoding.
- `main.py` Contains the user interface.

## Achieved Time and Space Complexities
The time complexity of neural networks in the training phase is typically O(EBN^2), where E is the number of epochs, B is the batch size, and N is the number of neurons in the network. Based on code analysis, my algorithm implements this time complexity. Space complexity is affected by the number of network parameters, i.e., weights and biases, as well as the number and size of inputs. The algorithm's space complexity is thus O(BN+BD+N^2), where B is the batch size, N is the number of neurons, and D is the number of features in each sample.

## Possible Shortcomings and Suggestions for Improvement
The user interface could still be improved. Hyperparameters could be optimized, and the depth or width of the neural network increased. Images could also be preprocessed, for example, by rotating them. These could lead to better accuracies, but the current implementation has nevertheless achieved quite good results.

## Use of Large Language Models
ChatGPT has been used as an aid in the project for:
- Information retrieval
- Debugging error messages
- Explaining some concepts
- Improving the spelling and clarity of documents
No code has been produced using language models.
