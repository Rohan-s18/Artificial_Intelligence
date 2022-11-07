#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:01:57 2022

@author: rohansingh
"""

# Python Module to find the posterior of different hypothesis using Bayesian Learning

# Imports
import plotly.express as px
import numpy as np
from math import comb
import matplotlib.pyplot as plt
import pandas as pd
import random
import plotly.graph_objects as go

import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'


# Creating a dataset of length 100
# 1: lime
# 0: cherry

d = []
observations = []
d.append("-")

random.seed(5)

for i in range (0,100,1):
    d.append(random.randint(0,1))
    observations.append(i)
    
 
#For h1
def getH1():
    prior = 0.1 #given in the book
    cherry = 1  
    lime = 0

    h1_posterior = []
    h1_posterior.append(prior)

    numerator = prior

    h1_denom = 0.1
    h2_denom = 0.2
    h3_denom = 0.4
    h4_denom = 0.2
    h5_denom = 0.1
    
    for i in range (1,101,1):
        if(d[i] == 0):                #Cherry
            numerator *= cherry
            h1_denom *= 1
            h2_denom *= 0.75
            h3_denom *= 0.5
            h4_denom *= 0.25
            h5_denom *= 0
        else:                         #Lime
            numerator *= lime
            h5_denom *= 1
            h4_denom *= 0.75
            h3_denom *= 0.5
            h2_denom *= 0.25
            h1_denom *= 0
        denominator = h1_denom + h2_denom + h3_denom + h4_denom + h5_denom
        temp = numerator/denominator
        h1_posterior.append(temp)
    
    return h1_posterior


#For h2
def getH2():
    prior = 0.2 #given in the book
    cherry = 0.75
    lime = 0.25

    h2_posterior = []
    h2_posterior.append(prior)

    numerator = prior

    h1_denom = 0.1
    h2_denom = 0.2
    h3_denom = 0.4
    h4_denom = 0.2
    h5_denom = 0.1
    
    for i in range (1,101,1):
        if(d[i] == 0):                #Cherry
            numerator *= cherry
            h1_denom *= 1
            h2_denom *= 0.75
            h3_denom *= 0.5
            h4_denom *= 0.25
            h5_denom *= 0
        else:                         #Lime
            numerator *= lime
            h5_denom *= 1
            h4_denom *= 0.75
            h3_denom *= 0.5
            h2_denom *= 0.25
            h1_denom *= 0
        denominator = h1_denom + h2_denom + h3_denom + h4_denom + h5_denom
        temp = numerator/denominator
        h2_posterior.append(temp)
    
    return h2_posterior



#For h3
def getH3():
    prior = 0.4 #given in the book
    cherry = 0.5
    lime = 0.5

    h3_posterior = []
    h3_posterior.append(prior)

    numerator = prior

    h1_denom = 0.1
    h2_denom = 0.2
    h3_denom = 0.4
    h4_denom = 0.2
    h5_denom = 0.1
    
    for i in range (1,101,1):
        if(d[i] == 0):                #Cherry
            numerator *= cherry
            h1_denom *= 1
            h2_denom *= 0.75
            h3_denom *= 0.5
            h4_denom *= 0.25
            h5_denom *= 0
        else:                         #Lime
            numerator *= lime
            h5_denom *= 1
            h4_denom *= 0.75
            h3_denom *= 0.5
            h2_denom *= 0.25
            h1_denom *= 0
        denominator = h1_denom + h2_denom + h3_denom + h4_denom + h5_denom
        temp = numerator/denominator
        h3_posterior.append(temp)
    
    return h3_posterior


#For h4
def getH4():
    prior = 0.2 #given in the book
    cherry = 0.25
    lime = 0.75

    h4_posterior = []
    h4_posterior.append(prior)

    numerator = prior

    h1_denom = 0.1
    h2_denom = 0.2
    h3_denom = 0.4
    h4_denom = 0.2
    h5_denom = 0.1
    
    for i in range (1,101,1):
        if(d[i] == 0):                #Cherry
            numerator *= cherry
            h1_denom *= 1
            h2_denom *= 0.75
            h3_denom *= 0.5
            h4_denom *= 0.25
            h5_denom *= 0
        else:                         #Lime
            numerator *= lime
            h5_denom *= 1
            h4_denom *= 0.75
            h3_denom *= 0.5
            h2_denom *= 0.25
            h1_denom *= 0
        denominator = h1_denom + h2_denom + h3_denom + h4_denom + h5_denom
        temp = numerator/denominator
        h4_posterior.append(temp)
    
    return h4_posterior


#For h5
def getH5():
    prior = 0.1 #given in the book
    cherry = 0
    lime = 1

    h5_posterior = []
    h5_posterior.append(prior)

    numerator = prior

    h1_denom = 0.1
    h2_denom = 0.2
    h3_denom = 0.4
    h4_denom = 0.2
    h5_denom = 0.1
    
    for i in range (1,101,1):
        if(d[i] == 0):                #Cherry
            numerator *= cherry
            h1_denom *= 1
            h2_denom *= 0.75
            h3_denom *= 0.5
            h4_denom *= 0.25
            h5_denom *= 0
        else:                         #Lime
            numerator *= lime
            h5_denom *= 1
            h4_denom *= 0.75
            h3_denom *= 0.5
            h2_denom *= 0.25
            h1_denom *= 0
        denominator = h1_denom + h2_denom + h3_denom + h4_denom + h5_denom
        temp = numerator/denominator
        h5_posterior.append(temp)
    
    return h5_posterior



h1_posterior = getH1()
h2_posterior = getH2()
h3_posterior = getH3()
h4_posterior = getH4()
h5_posterior = getH5()

#Creating the Dataframes for the hypothesis
df_1 = pd.DataFrame((list(zip(d,h1_posterior))),columns=["Data","Posterior for h1"])
df_2 = pd.DataFrame((list(zip(d,h2_posterior))),columns=["Data","Posterior for h2"])
df_3 = pd.DataFrame((list(zip(d,h3_posterior))),columns=["Data","Posterior for h3"])
df_4 = pd.DataFrame((list(zip(d,h4_posterior))),columns=["Data","Posterior for h4"])
df_5 = pd.DataFrame((list(zip(d,h5_posterior))),columns=["Data","Posterior for h5"])

#Plotting the posteriors
fig = go.Figure()
fig.add_trace(go.Scatter(x = observations, y = h1_posterior, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_posterior, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_posterior, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_posterior, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_posterior, mode='lines', name='p(h5|d)'))

fig.show()

#Finding the probability of the next candy being lime
prediction = []

#P(X is lime) = P(lime|h1)P(h1|d) + ..... + P(lime|h5)P(h5|d)
for i in range(0,100,1):
    temp = 0.25*h2_posterior[i]
    temp += 0.5*h3_posterior[i]
    temp += 0.75*h4_posterior[i]
    temp += 1*h5_posterior[i]
    prediction.append(temp)
    
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x = observations, y = prediction, mode='lines', name='p(Xn+1|dn)'))
fig_pred.update_layout( title="Probability that the next one is lime", xaxis_title="Number of Observations", yaxis_title="Prediction Probability")
fig_pred.show()




















