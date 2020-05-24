# imports
import numpy as np


# sigmoid function to convert to non linear function
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + (np.exp(-x)))


# input data
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# output data
y = np.array([[0, 0, 1, 1]]).T

# initial weights
np.random.seed(1)
syn0 = 2 * np.random.random((3, 1)) - 1

# training
for iter in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    l1_error = y - l1

    # multiply error by the slop of sigmoid
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print('Output after training:', l1)
