"""
Author: Rohan Singh
Python Module to find the intersectio between drugs across ic50 and dd disease datasets
"""

#  Imports
import pandas as pd
import numpy as np
from pubchempy import *
import random


#  Getting the attributes for ic50
def get_ic(filepath, attr):
    df = pd.read_csv(filepath)
    temp = df[attr].to_numpy()
    return temp


#  Getting the attributes for dd disease
def get_ddd(filepath, attr):
    df = pd.read_csv(filepath)
    temp = df[attr].to_numpy()
    return temp


#  helper function to get the intersection between the kinome names and the ddi names for drugs
def get_intersection_drugs(db1, db2):
    db1_set  = set(db1)
    db2_set = set(db2)
    intersections = db1_set.intersection(db2_set)
    print(len(intersections))
    return np.array(intersections), intersections


#  Function to get the intersection dataframe for ddd
def intersect_ddd(intersection, filepath):
    pass


#  Function to get the intersection dataframe for ic50
def intersect_ic50(intersection, filepath):
    df = pd.read_csv(filepath)
    
    names = df["CELL_NAME"].to_numpy()
    avg = df["AVERAGE"].to_numpy()

    avgs = []

    for i in range(0,len(names),1):
        if names[i] in intersection:
            avgs.append(avg[i])
    
    return np.array(avgs)



#  Function to write it back to the csv
def flush_csv(arr, filepath, cols):
    df = pd.DataFrame(arr, columns=cols)
    df.to_csv(filepath)


#  Main Function
def main():

    filepath_ic50 = "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/IC50.csv"
    filepath_ddd = "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersection_pairs_dd_disease.csv"

    ic = get_ic(filepath_ic50,"CELL_NAME")
    ddd = get_ddd(filepath_ddd,"cell_line")

    intersection_arr, intersection_set = get_intersection_drugs(ic,ddd)

    print("The Intersection is:")
    print(intersection_set)

    print("\n\n")

    print(ic)
    print("\n\n")
    print(ddd)

    #ddd_intersect = intersect_ddd(intersection_set, filepath_ddd)

    ic50_intersect = intersect_ic50(intersection_set, filepath_ic50)


    #print(ddd_intersect)
    print("\n")

    print(ic50_intersect)
    print("\n")

    ddd_int_filepath = "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersections/ddd.csv"
    ic50_int_filepath = "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersections/ic50.csv"

    #flush_csv(ddd_intersect, ddd_int_filepath)
    #flush_csv(ic50_intersect, ic50_int_filepath, cols=["ic50"])





if __name__ == "__main__":
    main()
