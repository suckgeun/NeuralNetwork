import numpy as np
import random

class Network(object):
    """
    this class assumes that we have only 3 layered Neural Network. 
    So, we have one input layer, one hidden layer, and one output layer. 
    """

    def __init__(self, architecture):
        '''
        Initialize Network.
        
        as parameter, only gets three valued list, which is 
        [input layer, hidden layer, output layer] form. 
        
        1. Creates biases for hidden layer and output layer.
            Note that input layer doesn't have biases. 
        
        2. Creates weights for (input-hidden), (hidden-output) connections. 
        
        
        :param architecture: list
        :type architecture: Architecture of the neural network. 
                            Must be [ number of units in input layer,
                                       number of units in hidden layer, 
                                       number of units in output layer ] 
        '''
        self.num_layers = len(architecture)
        self.architecture = architecture
        
        nnode_input = architecture[0]
        nnode_hidden = architecture[1]
        nnode_output = architecture[2]
        
        self.biases = [np.random.randn(nnode_hidden, 1), 
                       np.random.randn(nnode_output, 1)]
        
        self.weights = [np.random.randn(nnode_hidden, nnode_input), 
                        np.random.randn(nnode_output, nnode_hidden)]
    
    
    def SGD(self, training_data, num_epochs, mini_batch_size, learn_rate,
            test_data=None):
        '''
        Train the neural network using mini-batch stochastic gradient descent. 
        
        If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        
        
        :param training_data: list of tuples "(x, y)" representing the 
                        training inputs and the desired outputs. 
        :type training_data: list of tuples
        :param num_epochs: number of epochs to train 
        :type num_epochs: int
        :param mini_batch_size: mini batch size
        :type mini_batch_size: int 
        :param learn_rate: learning rate
        :type learn_rate: float
        :param test_data: testing data
        :type test_data: 
        '''        
        for i in xrange(num_epochs):
            
            mini_batches = self.create_mini_batches(
                    training_data, mini_batch_size)
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learn_rate)
            
            if test_data is not None:
                num_test = len(test_data)
                num_corret = self.evaluate(test_data)
                print "Epoch {0}: {1} / {2}".format(i, num_corret, num_test)
            else:
                print "Epoch {0} complete".format(i)
                
    def create_mini_batches(self, data, size):
        """Creates Mini batches given data and size"""
        return [ data[i:i+size] for i in xrange(0, len(data), size) ]
    
    
    def update_mini_batch(self, mini_batch, learn_rate):
        '''
        Update the network's weights and biases. 
        
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        
        :param mini_batch: mini batch
        :type mini_batch: list of tuples ``(x, y)``
        :param learn_rate: learning rate
        :type learn_rate: float 
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learn_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learn_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    
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




