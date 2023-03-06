"""
Author: Rohan Singh
3/3/2023
Code for various types of Matrix Products
"""


#  Imports
import numpy as np
import random as rand
import numpy.linalg as lin



#  Function to create a random numpy matrix
def create_random_matrix_numpy(m,n,min,max)->np.ndarray:
    matrix = []
    for i in range (0,m,1):
        temp = np.zeros(n)
        for j in range(0,n,1):
            temp[j] = (min+int((max-min)*rand.random()))
        matrix.append(temp)
    return np.array(matrix)


#  Function to create a random matrix in list form
def create_random_matrix_list(m,n, min, max)->list:
    matrix = []
    for i in range (0,m,1):
        temp = []
        for j in range(0,n,1):
            temp.append(min+int((max-min)*rand.random()))
        matrix.append(temp)
    return matrix



#  Main function
def main():
    
    print("\n")
    #Creating Matrices
    A = create_random_matrix_numpy(5,5,-10,10)
    print("Matrix A is: ",A,"\n")
    B = create_random_matrix_numpy(5,5,-10,10)
    print("Matrix B is: ",B,"\n")


    print("\n")
    C = create_random_matrix_list(5,5,-100,100)
    print("Matrix C is: ", C, "\n")
    D = create_random_matrix_list(5,5,-100,100)
    print("Matrix D is: ", D, "\n")


    #Multiplying 2 matrices
    print("The product of A and B is: \n",np.matmul(A,B))

    print("\n")

    #Getting the power of 2 matrices
    print("The power of a matrix A to 5 is: \n",lin.matrix_power(A,5))

    print("\n")

if __name__ == "__main__":
    main()
