#initialising weights for the feed forward neural network

import numpy as np

# Inputs
x1 = 5
x2 = 90
x3 = 80

# Random weights
w1 = np.random.rand()
w2 = np.random.rand()
w3 = np.random.rand()

# Random bias
b = np.random.rand()

print("Weight 1:", w1)
print("Weight 2:", w2)
print("Weight 3:", w3)
print("Bias:", b)

# Weighted Sum
output = (x1*w1) + (x2*w2) + (x3*w3) + b

print("Network Output:", output)
