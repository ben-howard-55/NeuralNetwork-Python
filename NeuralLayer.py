import numpy as np

# This python file represents the basic logic behind a Layer in a Neural Network.
# This Layer has 3 nodes and 4 nodes in the previous Layer.

# inputs - represent values from previous Layer or sensors.
inputs = [1, 2, 3, 2.5]

# weights - represent individual weights connecting between each previous node and the current node.
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# biases - represents the additional value added or removed from each node's final value
biases = [2, 3, 0.5]

# output - represents the final value of each node in the current layer
output = np.dot(weights, inputs) + biases

print(output)
