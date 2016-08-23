import numpy as np
import random

class Network(object):
    """
    this class assumes that we have only 3 layered Neural Network. 
    So, we have one input layer, one hidden layer, and one output layer. 
    """

    def __init__(self, architecture):
        """
        Initialize Network.

        as parameter, only gets three valued list, which is 
        [input layer, hidden layer, output layer] form. 
        
        Creates biases for hidden layer and output layer.
        Note that input layer doesn't have biases. 
        
        Creates weights for (input-hidden), (hidden-output) connections. 

        @type  architecture: list 
        @param architecture: Architecture of the neural network. 
                             Must be [ number of units in input layer,
                                       number of units in hidden layer, 
                                       number of units in output layer ] 
        """
        self.num_layers = len(architecture)
        self.architecture = architecture
        
        nnode_input = architecture[0]
        nnode_hidden = architecture[1]
        nnode_output = architecture[2]
        
        self.biases = [np.random.randn(nnode_hidden, 1), \
                       np.random.randn(nnode_output, 1)]
        
        self.weights = [np.random.randn(nnode_hidden, nnode_input), \
                        np.random.randn(nnode_output, nnode_hidden)]


    def feedforword(self, a):
        """Return the output of network."""
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

def sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    """Derivative of the sigmoid function"""
    return sigmoid(x)*(1-sigmoid(x))


