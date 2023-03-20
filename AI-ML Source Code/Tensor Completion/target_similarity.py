"""
Author: Rohan Singh
Python Module for finding creating a similarity matrix between two matrices
The Similrity distance will be calculated based on the target site on which the drugs work on a disease
"""


#  Imports
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import pandas as pd


#  Helper function to retrieve the data and store it into a dataframe
def create_tensor():
    file = ""
    df = pd.read_csv(file)
    



#  Helper function to get the 2-norm similarity between the 2 vectors
def get_distance(vec1, vec2):
    y = np.dot(vec1,vec2)
    a = np.linalg.norm(vec1,2)
    b = np.linalg.norm(vec2,2)
    z = y/(a*b)
    return z


#  Helper function to get the similarity between the 2 vectors (custom)
def get_special_distance(vec1, vec2):
    y = np.dot(vec1,vec2)
    a = np.linalg.norm(vec1,2)
    b = np.linalg.norm(vec2,2)
    z = y/(a*b)
    return z


#  Main function
def main():
    pass


if __name__ == "__main__":
    main()