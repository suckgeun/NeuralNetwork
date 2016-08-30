import numpy as np 
import random

class Network(object):
    
    def __init__(self, arch = [2, 3, 2]):
        
        self.arch = arch
        
        self.biases = np.array([np.random.randn(nnode, 1) for nnode in arch[1:]])
        
        self.weights = np.array([
            np.random.randn(nnode, nweight) for nnode, nweight
            in zip(arch[1:], arch[:-1])])
                
    def fit(self, training_data, testing_data,
            num_epochs = 30, batch_size = 10, learn_rate = 0.05):
        
        random.shuffle(training_data)
        
        for i in range(num_epochs):
            
            batches = [training_data[k : k + batch_size] 
                           for k in xrange(0, len(training_data), batch_size)]
            
            for batch in batches:
                sum_del_w = None
                sum_del_b = None
                for data, label in batch:
                    xs, ys = self.feedforward(data)
                    errors = self.calc_errors(ys[-1], label)
                    del_w, del_b = self.calc_delta_weight_bias(xs, ys, errors)
                    
                    if sum_del_w is None:
                        sum_del_w = del_w
                        sum_del_b = del_b
                    else:
                        sum_del_w += del_w
                        sum_del_b += del_b                    
                
                self.weights += learn_rate*sum_del_w
                self.biases += learn_rate*sum_del_b
                
            self.predict(testing_data, i)
    
    def feedforward(self, a):
        xs = []
        ys = []
        for weight, bias in zip(self.weights, self.biases):
            xs.append(a)
            a = sigmoid(np.dot(weight, a) + bias)
            ys.append(a)
        return np.array(xs), np.array(ys)
    
    def calc_errors(self, y, label):
        errors = []
        error = label - y 
        errors.append(error)
        for weight in reversed(self.weights[1:]):
            error = np.dot(weight.T, error)
            errors.insert(0, error)
        return errors
    
    def calc_delta_weight_bias(self, xs, ys, errors):
        delta_w = []
        delta_b = []
        for x, y, e in zip(xs, ys, errors):
            sigmoid_prime = y*(1-y)
            delta_w.append(np.dot(e*sigmoid_prime, x.T))
            delta_b.append(np.array(e*sigmoid_prime, ndmin=2))
        return np.array(delta_w), np.array(delta_b)
    
    def predict(self, testing_data, index):
        count = 0
        for data, label in testing_data:
            xs, ys = self.feedforward(data)
            
            if np.argmax(ys[-1]) == label:
                count += 1
                
        print "index: {0}, count: {1}".format(index, count)
    
def sigmoid(x):
    return 1/(1+np.exp(-x))















