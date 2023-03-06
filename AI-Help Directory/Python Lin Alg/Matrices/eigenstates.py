"""
Author: Rohan Singh
3/3/23
Python module to demonstrate getting eigenstates and eigenvalues for a given square matrix
"""


#  Imports
import numpy as np
import random as rand
import numpy.linalg as lin


#  Function to create a random numpy square matrix
def create_random_square_matrix(n,min,max)->np.ndarray:
    matrix = []
    for i in range (0,n,1):
        temp = np.zeros(n)
        for j in range(0,n,1):
            temp[j] = (min+int((max-min)*rand.random()))
        matrix.append(temp)
    return np.array(matrix)


#  Main function
def main():
    
    #Creating the matrix
    A = create_random_square_matrix(5,-10,10)
    print("\n")
    print("The Matrix A is:\n",A,"\n\n")

    #Getting the eigenstates and eigenvalues of A
    eigenvalues, eigenstates = lin.eig(A)

    print("The eigenvalues of A are:\n")
    for i in range(0,len(eigenvalues),1):
        print(eigenvalues[i])

    print("\nThe eigenstates of A are:\n")
    for i in  range(0,len(eigenstates),1):
        print(eigenstates[i])

    print("\n")

if __name__ == "__main__":
    main()