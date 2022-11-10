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

import sys


# Creating a dataset of length 100
# 1: lime
# 0: cherry

seed = 5

if(len(sys.argv) > 1):
    seed = sys.argv[1]

d = []
observations = []
d.append("-")

random.seed(seed)

for i in range (0,100,1):
    observations.append(i)
    
def makeDataSet(hyp):
    dataset = []
    if(hyp == "h1"):
        for i in range (0,100,1):
            dataset.append(0)
    elif(hyp == "h2"):
        for i in range (0,75,1):
            dataset.append(0)
        for i in range (0,25,1):
            dataset.append(1)
    elif(hyp == "h3"):
        for i in range (0,50,1):
            dataset.append(0)
        for i in range (0,50,1):
            dataset.append(1)
    elif(hyp == "h4"):
        for i in range (0,75,1):
            dataset.append(1)
        for i in range (0,25,1):
            dataset.append(0)
    elif(hyp == "h5"):
        for i in range (0,100,1):
            dataset.append(1)
    else:
        return 
    arr = np.array(dataset)
    np.random.shuffle(arr)
    dataset = arr.tolist()
    ds = []
    ds.append("-")
    for i in range (0,100,1):
        ds.append(dataset[i])
    return ds
    
 
#For h1
def getH1(candy):
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
        if(candy[i] == 0):                #Cherry
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
def getH2(candy):
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
        if(candy[i] == 0):                #Cherry
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
def getH3(candy):
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
        if(candy[i] == 0):                #Cherry
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
def getH4(candy):
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
        if(candy[i] == 0):                #Cherry
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
def getH5(candy):
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
        if(candy[i] == 0):                #Cherry
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


#Finding the probability of the next candy being lime
def getNextPrediction(h2_p, h3_p, h4_p, h5_p):
    prediction = []

    #P(X is lime) = P(lime|h1)P(h1|d) + ..... + P(lime|h5)P(h5|d)
    for i in range(0,100,1):
        temp = 0.25*h2_p[i]
        temp += 0.5*h3_p[i]
        temp += 0.75*h4_p[i]
        temp += 1*h5_p[i]
        prediction.append(temp)
    return prediction


#For H1 Observations
temp = makeDataSet("h1")

h1_p = getH1(temp)
h2_p = getH2(temp)
h3_p = getH3(temp)
h4_p = getH4(temp)
h5_p = getH5(temp)

#Plotting the posteriors
fig = go.Figure()
fig.add_trace(go.Scatter(x = observations, y = h1_p, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_p, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_p, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_p, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_p, mode='lines', name='p(h5|d)'))
fig.update_layout( title="Posterior Probability when the observations are from H1", xaxis_title="Number of Observations", yaxis_title="Posterior Probability for the Hypotheses")
fig.show()


pred = getNextPrediction(h2_p, h3_p, h4_p, h5_p)
    
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x = observations, y = pred, mode='lines', name='p(Xn+1|dn)'))
fig_pred.update_layout( title="Probability that the next one is lime for H1 observations", xaxis_title="Number of Observations", yaxis_title="Prediction Probability")
fig_pred.show()

#For H2 Observations
temp = makeDataSet("h2")

h1_p = getH1(temp)
h2_p = getH2(temp)
h3_p = getH3(temp)
h4_p = getH4(temp)
h5_p = getH5(temp)

#Plotting the posteriors
fig = go.Figure()
fig.add_trace(go.Scatter(x = observations, y = h1_p, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_p, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_p, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_p, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_p, mode='lines', name='p(h5|d)'))
fig.update_layout( title="Posterior Probability when the observations are from H2", xaxis_title="Number of Observations", yaxis_title="Posterior Probability for the Hypotheses")
fig.show()


pred = getNextPrediction(h2_p, h3_p, h4_p, h5_p)
    
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x = observations, y = pred, mode='lines', name='p(Xn+1|dn)'))
fig_pred.update_layout( title="Probability that the next one is lime for H2 observations", xaxis_title="Number of Observations", yaxis_title="Prediction Probability")
fig_pred.show()

#For H1 Observations
temp = makeDataSet("h3")

h1_p = getH1(temp)
h2_p = getH2(temp)
h3_p = getH3(temp)
h4_p = getH4(temp)
h5_p = getH5(temp)

#Plotting the posteriors
fig = go.Figure()
fig.add_trace(go.Scatter(x = observations, y = h1_p, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_p, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_p, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_p, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_p, mode='lines', name='p(h5|d)'))
fig.update_layout( title="Posterior Probability when the observations are from H3", xaxis_title="Number of Observations", yaxis_title="Posterior Probability for the Hypotheses")
fig.show()


pred = getNextPrediction(h2_p, h3_p, h4_p, h5_p)
    
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x = observations, y = pred, mode='lines', name='p(Xn+1|dn)'))
fig_pred.update_layout( title="Probability that the next one is lime for H3 observations", xaxis_title="Number of Observations", yaxis_title="Prediction Probability")
fig_pred.show()

#For H4 Observations
temp = makeDataSet("h4")

h1_p = getH1(temp)
h2_p = getH2(temp)
h3_p = getH3(temp)
h4_p = getH4(temp)
h5_p = getH5(temp)

#Plotting the posteriors
fig = go.Figure()
fig.add_trace(go.Scatter(x = observations, y = h1_p, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_p, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_p, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_p, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_p, mode='lines', name='p(h5|d)'))
fig.update_layout( title="Posterior Probability when the observations are from H4", xaxis_title="Number of Observations", yaxis_title="Posterior Probability for the Hypotheses")
fig.show()


pred = getNextPrediction(h2_p, h3_p, h4_p, h5_p)
    
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x = observations, y = pred, mode='lines', name='p(Xn+1|dn)'))
fig_pred.update_layout( title="Probability that the next one is lime for H4 observations", xaxis_title="Number of Observations", yaxis_title="Prediction Probability")
fig_pred.show()

#For H5 Observations
temp = makeDataSet("h5")

h1_p = getH1(temp)
h2_p = getH2(temp)
h3_p = getH3(temp)
h4_p = getH4(temp)
h5_p = getH5(temp)

#Plotting the posteriors
fig = go.Figure()
fig.add_trace(go.Scatter(x = observations, y = h1_p, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_p, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_p, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_p, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_p, mode='lines', name='p(h5|d)'))
fig.update_layout( title="Posterior Probability when the observations are from H5", xaxis_title="Number of Observations", yaxis_title="Posterior Probability for the Hypotheses")
fig.show()


pred = getNextPrediction(h2_p, h3_p, h4_p, h5_p)
    
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x = observations, y = pred, mode='lines', name='p(Xn+1|dn)'))
fig_pred.update_layout( title="Probability that the next one is lime for H5 observations", xaxis_title="Number of Observations", yaxis_title="Prediction Probability")
fig_pred.show()




















