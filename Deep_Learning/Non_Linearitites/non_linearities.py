"""
Author: Rohan Singh
Python Script with all non-linearities
"""

# imports
import numpy as np


# RELU
def relu(x):
    if x > 0:
        return x
    return 0


# Sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))


# Absolute Value
def abs(x):
    return np.abs(x)

