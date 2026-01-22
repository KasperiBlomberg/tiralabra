# Fashion MNIST Neural Network

## About the Project
This project implements a Multilayer Perceptron (MLP) neural network from scratch using Python and NumPy. Designed to classify clothing items from the Fashion MNIST dataset, the network utilizes a custom-built architecture with ReLU activations, Softmax output, and L2 regularization. It achieves ~89% accuracy on test data, demonstrating fundamental deep learning concepts without relying on high-level frameworks like TensorFlow or PyTorch.

### Technical Documentation & Quality Assurance
Implementation: For a detailed breakdown of the architecture, algorithms, and time/space complexity, please refer to the [Implementation Document](./docs/implementation_document.md).

Testing: The codebase is fully validated with unit and integration tests to ensure mathematical correctness. Detailed coverage reports are available in the [Testing Document](./docs/test_document.md).
## User Guide
### Installation
First, download or clone the repository to your local machine.
Navigate to the project directory and install the dependencies using Poetry:
```bash
   poetry install
   ```
Activate the virtual environment:
```bash
   poetry shell
   ```
### Running the Application
Start the program with the following command:
```bash
   python3 src/main.py
   ```
### Usage
**Training:** The neural network must be trained before use. Select option 1 from the menu. You can simply press Enter to use the optimized default hyperparameters. Please note that training may take a few minutes.

**Evaluation:** Once trained, select option 2 to test the network against the test dataset and view the total accuracy percentage.

**Prediction:** To view individual images and see if the network classifies them correctly, select option 3.

## Suomenkieliset dokumentit
[Määrittelydokumentti](./docs/maarittelydokumentti.md)

[Toteutusdokumentti](./docs/toteutusdokumentti.md)

[Testausdokumentti](./docs/testausdokumentti.md)
