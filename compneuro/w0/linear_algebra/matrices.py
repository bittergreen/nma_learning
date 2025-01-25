import numpy as np

from utils import *


# Create P (using np array)
P = np.array([[1, 3], [2, 1]])

# Create g_p (using np array)
g_p = np.array([16, 7])

# Solve for r (using np.linalg.inv)
r = np.linalg.inv(P) @ g_p

# Print r
print(r)


# @markdown Execute to visualize linear transformations
P = np.array([[1, 3], [2, 1]])
plot_linear_transformation(P, name='p')


Q = np.array([[4, 1], [8, 2]])
plot_linear_transformation(Q, name='q')

print(np.linalg.matrix_rank(Q))

print(np.linalg.eig(P))

