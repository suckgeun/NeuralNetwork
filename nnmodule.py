import numpy as np

class Network(object):
    """
    this class assumes that we have only 3 layered Neural Network. 
    So, we have one input layer, one hidden layer, and one output layer. 
    """

    def __init__(self, nn_arch):
        """
        @type  nn_arch: list 
        @param nn_arch: Architecture of the neural network. Must be [x, y, z] form, where x,y,z are integers, 
                        x: number of units in input layer
                        y: number of units in hidden layer
                        z: number of units in output layer
        """
        self.num_layers = 3
        self.nn_arch = nn_arch
        # get the list [num_units_in_hidden_layer, num_unts_in_output_layer]
        hidden_and_output_layers = nn_arch[1:]
        self.biases = [np.random.randn(num_units, 1) for num_units in hidden_and_output_layers]
        # get the list [(num_units_input_layer, num_units_hidden_layer), (num_units_hidden_layer, num_units_output_layer)]
        connections_betw_layers = zip(nn_arch[:-1], nn_arch[1:])
        self.weights = [np.random.randn(end_units, start_units) for start_units, end_units in connections_betw_layers]

