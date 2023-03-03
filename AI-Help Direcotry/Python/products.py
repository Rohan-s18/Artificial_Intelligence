"""
Author: Rohan Singh
3/2/2023
Python Module to demonstrate various vector products in python using numpy
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


#  Main function
def main():

    print("\n")

    #dot-product
    x = create_random_array(10,-100,100)
    y = create_random_array(10,-100,100)
    print("The vector x is: ", x)
    print("The vector y is: ", y)
    print("The dot prodcuct of x and y is: ",np.dot(x,y))

    print("\n")

    #cross-product
    v = create_random_vector(3,-100,100)
    u = create_random_vector(3,-100,100)
    print("The vector v is: ", v)
    print("The vector u is: ", u)
    print("The cross product of v and u is: ",np.cross(v,u))

    print("\n")

    #outer-product
    w = create_random_array(10,-100,100)
    z = create_random_array(10,-100,100)
    print("The vector w is: ", w)
    print("The vector z is: ", z)
    print("The outer-product of w and z is:",np.outer(w,z))

    print("\n")


if __name__ == "__main__":
    main()