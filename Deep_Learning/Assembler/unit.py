"""
Author: Rohan Singh
Python Script for coding a single unit in a neural network layer
"""

# imports
import numpy as np
from Non_Linearitites.non_linearities import *



class Unit():

    def __init__(self, input_dimension, output_dimension, non_linearity) -> None:
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.weights = np.ones(input_dimension)
        self.non_linearity = non_linearity

    
    def set_weights(self, weights):
        if len(weights) == self.input_dimension:
            self.weights = weights

    def get_weights(self):
        return self.weights
    
    def predict(self, data):
        z = np.dot(self.weights, data)

        if (self.non_linearity=="RELU"):
            return relu(z)
        
        elif (self.non_linearity=="SIGMOID"):
            return sigmoid(z)
        
        elif (self.non_linearity=="ABSOLUTE VALUE"):
            return abs(z)
        
        else:
            return z
        
    def forward(self, data):
        output = []
        for i in range(0,self.output_dimension, 1):
            output.append(self.predict(data=data))
        return np.array(output)
        




