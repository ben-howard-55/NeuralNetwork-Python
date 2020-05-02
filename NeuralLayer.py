import numpy as np

# This python file represents the basic logic behind a Layer in a Neural Network.
# This Layer has 3 nodes and 4 nodes in the previous Layer.

# inputs - represent values from previous Layers or sensors.
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# weights - represent individual weights connecting between each previous node and the current node.
weights1 = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
        
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, 0.5]

# biases - represents the additional value added or removed from each node's final value
biases = [2, 3, 0.5]

# output - represents the final value of each node in the current layer
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)
