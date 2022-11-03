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
from sklearn import datasets


def getData():
    # Importing the iris data set
    iris = datasets.load_iris()
    
    # Extracting the data for petal length and width
    X = iris.data[:, 2:]
    return X

def plotIris():
    iris = datasets.load_iris()
    X = iris.data[:, 2:]
    y = iris.target
    mtp.scatter(X[:, 0],X[:, 1],c=y)
    mtp.xlabel("Petal Length")
    mtp.ylabel("Petal Width")
    mtp.show()
