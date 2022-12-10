#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:41:05 2022

@author: rohansingh
"""

# This module can be used to find the likelihood in Bernoulli's trials

#Imports
import plotly.express as px
from math import comb
import pandas as pd

import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

#Array Representing the possible values of 'y'
y = [0,1,2,3,4]

#Getting the likelihoods
likelihood = []

theta = 0.75
n = 4

#Iterating through each value of 'y' and storing the likelihood value to the corresponding index
for i in range (0,5,1):
    # Getting the combination factor
    temp = comb(4,i) 
    temp *= (theta**i)
    temp *= ((1-theta)**(n-i))
    likelihood.append(temp)
    
    
#Plotting the Data
df = pd.DataFrame(list(zip(y,likelihood)), columns = ["y","likelihood"])

fig = px.scatter(df, x="y", y="likelihood")
fig.show()