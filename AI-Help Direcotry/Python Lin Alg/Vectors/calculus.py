"""
Author: Rohan Singh
3/3/2023
Python Module to demonstrate vector calculus in python using numpy
"""

#  Imports
import numpy as np
import random as rand


#  Function to create a random numpy array
def create_random_array(n,min,max)->np.ndarray:
    temp = np.zeros(10)
    for i in range(0,n,1):
        temp[i] = (min+int((max-min)*rand.random()))
    return temp


#  Function to create a random list
def create_random_vector(n, min, max)->list:
    temp = []
    for i in range(0,n,1):
        temp.append(min+int((max-min)*rand.random()))
    return temp


#  Function to create random infinitesimal elements
def create_random_element(n)->np.ndarray:
    temp = np.zeros(10)
    for i in range(0,n,1):
        temp[i] = (rand.random()*0.1)
    return temp


#  Main function
def main():

    print("\n")

    #vector derivative for constant 'dx'
    dx = 0.1
    y = create_random_array(10,-100,100)
    print("dx is: ",dx)
    print("y is: ",y)
    print("dy/dx is: ",np.diff(y)/dx)

    print("\n")

    #vector derivative for non-constant 'dx'
    dx = create_random_element(10)
    y = create_random_array(10,-100,100)
    print("dx is: ",dx)
    print("y is: ",y)
    print("dy/dx is: ",np.diff(y)/np.diff(dx))

    print("\n")




if __name__ == "__main__":
    main()

