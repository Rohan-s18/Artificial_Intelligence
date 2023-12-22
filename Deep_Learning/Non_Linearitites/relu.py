"""
Author: Rohan Singh
Source Code for RELU non-linearity
"""

# Imports
import numpy as np

def relu(x):
    if x > 0:
        return x
    return 0