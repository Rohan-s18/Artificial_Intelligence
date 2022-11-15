#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:50:08 2022

@author: rohansingh
"""

# Source code for K-means clustering

#Getting imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

#this is used to get the iris data
df = pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")


centroids = np.array([10,78,130])
clusters = []
for i in range(0,150,1):
    clusters.append("-")

#Converting df to a 4 numpy arrays:
s_len = df["sepal_length"]
s_wid = df["sepal_width"]
p_len = df["petal_length"]
p_wid = df["petal_width"]

s_len = s_len.to_numpy()
s_wid = s_wid.to_numpy()
p_len = p_len.to_numpy()
p_wid = p_wid.to_numpy()

# Helper function to get the euclidean distance 
def euc_dist(i, j):
    dist_sum = 0
    dist_sum += np.power((s_len[i]-s_len[j]),2)
    dist_sum += np.power((s_wid[i]-s_wid[j]),2)
    dist_sum += np.power((p_len[i]-p_len[j]),2)
    dist_sum += np.power((p_wid[i]-p_wid[j]),2)
    return np.sqrt(dist_sum)

#Helper function for the objective function
def obj_funct_classifier(i,centroids,cluster):
    cl = 0
    dist = 10000000
    for j in range (0,len(centroids),1):
        new_dist = euc_dist(i,centroids[j])
        if(new_dist < dist):
            dist = new_dist
            cl = j
    cluster[i] = cl
    return dist

#The Objective function
def objectiveFunction(centroids,clusters):
    val = 0
    x = 0
    for i in range (0,len(clusters),1):
        x = obj_funct_classifier(i,centroids,clusters)
        val += x
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

