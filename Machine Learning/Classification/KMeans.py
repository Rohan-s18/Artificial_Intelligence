#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:20:04 2022

@author: rohansingh
"""

# Source code for K-means clustering

#Getting imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

#this is used to get the iris data
df = pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")

def plotIris():
    X = df.data[:, 3:]
    y = df.target
    mtp.scatter(X[:, 0],X[:, 1],c=y)
    mtp.xlabel("Petal Length")
    mtp.ylabel("Petal Width")
    mtp.show()
    
def eucDist(x1, x2, y1, y2):
    return np.sqrt( (np.power((x2-x1),2)) + (np.power((y2-y1),2)) )

def objFunction(mu, pos1, pos2):
    return 0

def calcKMeansIris(k):
    return 0
