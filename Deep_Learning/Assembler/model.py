"""
Author: Rohan Singh
Python Script to assemble a model
"""

# imports
import numpy as np
from layer import Layer

# model class
class Model():

    def __init__(self, input_dimension, output_dimension) -> None:
        self.layer_info = [[input_dimension, output_dimension]]

    def add_layer(self, input_dimension, output_dimension):
        pass