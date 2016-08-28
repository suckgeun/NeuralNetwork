import numpy as np
import neuralnet

# prepare data for Xor 
train_data = np.array([[[0], [0]], 
                       [[1], [0]], 
                       [[0], [1]], 
                       [[1], [1]]])
# label for Xor
label_data = np.array([[0], 
                       [1], 
                       [1], 
                       [0]])

# initialize network
net = neuralnet.Network()
# train a model
net.fit(train_data, label_data)
# test the model
net.predict(train_data, label_data)