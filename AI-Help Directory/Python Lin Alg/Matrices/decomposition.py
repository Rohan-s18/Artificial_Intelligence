"""
Author: Rohan Singh
3/3/23
Python module to demonstrate the various types of decomposition
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
    A = create_random_matrix_numpy(3,5,-10,10)
    print("Matrix A is: ",A,"\n")

    """
    Other matrices to try it on
    B = create_random_matrix_numpy(5,5,-10,10)
    print("Matrix B is: ",B,"\n")


    print("\n")
    C = create_random_matrix_list(5,5,-100,100)
    print("Matrix C is: ", C, "\n")
    D = create_random_matrix_list(5,5,-100,100)
    print("Matrix D is: ", D, "\n")

    """

    print("\n")

    #Getting the qr factorization of the matrix
    q, r = lin.qr(A)
    print("The Q-R factorization of A is: \n")
    print("Q is:\n",q,"\n")
    print("R is:\n",r,"\n\n")

    u, s, vh = lin.svd(A)
    print("The Singular Value Decomposition of A is:\n")
    print("U is:\n",u,"\n")
    print("E is:\n",s,"\n")
    print("Vh is:\n",vh,"\n\n")


if __name__ == "__main__":
    main()