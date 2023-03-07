"""
Author: Rohan Singh
Python Module for finding creating a similarity matrix between two vectors
"""

#  Imports
import numpy as np
import matplotlib.pyplot as plt
import random as rand



#  Helper Function to create a random numpy square matrix
def create_random_square_matrix(n,min,max)->np.ndarray:
    matrix = []
    for i in range (0,n,1):
        temp = np.zeros(n)
        for j in range(0,n,1):
            temp[j] = (min+int((max-min)*rand.random()))
        matrix.append(temp)
    return np.array(matrix)


#  Helper function to get the 2-norm similarity between the 2 vectors
def get_distance(vec1, vec2):
    y = np.dot(vec1,vec2)
    a = np.linalg.norm(vec1,2)
    b = np.linalg.norm(vec2,2)
    z = y/(a*b)
    return z


#  Function to get the similarity matrix
def get_similarity(matrix1, matrix2):
    matrix = []
    for i in range(0,len(matrix1),1):
        list = []
        for j  in range(0,len(matrix2),1):
            list.append(get_distance(matrix1[i],matrix2[j]))
        matrix.append(np.array(list))
    return np.array(matrix)


#  Main Function
def main():
    
    #Creating the matrices
    A = create_random_square_matrix(5,-10,10)
    B = create_random_square_matrix(5,-10,10)
    print("\n")
    print("The Matrix A is:\n",A,"\n\n")
    print("The Matrix B is:\n",A,"\n\n")

    #Getting the similarity matrix between the same matrix
    S = get_similarity(A,A)
    print("\n")
    print("The Similarity Matrix for A and A is:\n",S,"\n\n")

    #Getting the similarity matrix between 2 different matrices
    S_2 = get_similarity(A,B)
    print("\n")
    print("The Similarity Matrix for A and B is:\n",S_2,"\n\n")



if __name__ == "__main__":
    main()

