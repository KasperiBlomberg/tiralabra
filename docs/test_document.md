## Unit Tests
Various helper functions have been tested in the `test_utils.py file`. This includes testing data loading and preprocessing. The activation functions used by the neural network are also covered. 
The functions are tested with relatively simple inputs, as the functions themselves are quite straightforward.

**Test Coverage Report**
![image](./coverage_screenshot.png)

Test coverage is nearly 100%; only the functionality for saving parameters to a file remains untested.

## Integration Testing
The testing of the neural network's core operation is implemented via integration testing. I did not find it useful to test components like the backpropagation algorithm in isolation; instead, I created an end-to-end test for it. 
I validated the network following the strategies outlined in this article using two comprehensive tests. I used a test_train function to verify that the network overfits to a small dataset (in this case, containing 10 samples). 
Simultaneously, I verified that all weights and biases are updated after every batch. I also verified that the training loss decreases after every iteration and that gradients are never zero. The test uses real data as input, ensuring the inputs are representative. 
Additionally, I used another test to verify that the order of samples does not affect the neural network's predictions. 

These tests ensure that all neural network methods function correctly.

## Running Tests
Tests can be executed in the virtual environment from the root directory with the command:
`pytest`

## Network's Performance
The network's strong performance also validates the correctness of the algorithm. I have achieved a peak accuracy of 88.9% on the test dataset. 
A similar MLP neural network linked in the Zalando repository (with layer sizes of 256, 128, and 100) achieved an accuracy of 88.3%. My model's efficiency is on par with this benchmark, suggesting the algorithm is functioning correctly.
