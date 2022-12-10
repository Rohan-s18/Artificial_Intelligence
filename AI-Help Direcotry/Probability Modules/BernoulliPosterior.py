#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 17:52:20 2022

@author: rohansingh
"""

#Imports
import plotly.express as px
from math import comb
import pandas as pd
import numpy as np

import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

# First function: Posterior for: H
# Second function: Posterior for: H,H
# Third function: Posterior for: H,H,T
# Fourth function: Posterior for: H,H,T,H

#First Function
def firstFunct():
    theta = np.linspace(0,1,10)
    theta = theta.tolist()
    f = []
    for i in range (0,10,1): #Calculating the posterior values
        temp = 2*theta[i]
        f.append(temp)
    df = pd.DataFrame(list(zip(theta,f)), columns = ["theta","posterior"])
    fig = px.scatter(df, x="theta", y="posterior")
    fig.show()
    
#Second Function
def secondFunct():
    theta = np.linspace(0,1,10)
    theta = theta.tolist()
    f = []
    for i in range (0,10,1): #Calculating the posterior values
        temp = theta[i]**2
        temp *= 3
        f.append(temp)
    df = pd.DataFrame(list(zip(theta,f)), columns = ["theta","posterior"])
    fig = px.scatter(df, x="theta", y="posterior")
    fig.show()
    
#Third Function
def thirdFunct():
    theta = np.linspace(0,1,10)
    theta = theta.tolist()
    f = []
    for i in range (0,10,1): #Calculating the posterior values
        temp = theta[i]**2
        temp *= 3
        f.append(temp)
    df = pd.DataFrame(list(zip(theta,f)), columns = ["theta","posterior"])
    fig = px.scatter(df, x="theta", y="posterior")
    fig.show()

#Fourth Function
def fourthFunct():
    theta = np.linspace(0,1,10)
    theta = theta.tolist()
    f = []
    for i in range (0,10,1): #Calculating the posterior values
        temp = theta[i]**3
        temp *= 20
        temp *= (1-theta[i])
        f.append(temp)
    df = pd.DataFrame(list(zip(theta,f)), columns = ["theta","posterior"])
    fig = px.scatter(df, x="theta", y="posterior")
    fig.show()


firstFunct()
secondFunct()
thirdFunct()
fourthFunct()












    