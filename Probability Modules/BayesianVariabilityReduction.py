#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 22:50:49 2022

@author: rohansingh
"""

#Imports
import plotly.express as px
import numpy as np
from math import comb
import matplotlib.pyplot as plt
import pandas as pd
import random
import plotly.graph_objects as go
import statistics
import plotly.io as pio

pio.renderers.default = 'browser'
observations = []
for i in range (0,100,1):
    observations.append(i)
   

"""========================================================================= Helper Functions are over here=========================================================================================================="""

    
#Helper method to create a shuffled bag based on the given type
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


"""===================================================================================================================================================================================================================================================="""
#Getting the Averages for h3

temp = makeDataSet("h3")
h1_p1 = getH1(temp)
h2_p1 = getH2(temp)
h3_p1 = getH3(temp)
h4_p1 = getH4(temp)
h5_p1 = getH5(temp)

#Single Bag
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Single bag of h3 candy")
fig.add_trace(go.Scatter(x = observations, y = h1_p1, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_p1, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_p1, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_p1, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_p1, mode='lines', name='p(h5|d)'))
fig.show()

temp = makeDataSet("h3")
h1_p2 = getH1(temp)
h2_p2 = getH2(temp)
h3_p2 = getH3(temp)
h4_p2 = getH4(temp)
h5_p2 = getH5(temp)

temp = makeDataSet("h3")
h1_p3 = getH1(temp)
h2_p3 = getH2(temp)
h3_p3 = getH3(temp)
h4_p3 = getH4(temp)
h5_p3 = getH5(temp)

temp = makeDataSet("h3")
h1_p4 = getH1(temp)
h2_p4 = getH2(temp)
h3_p4 = getH3(temp)
h4_p4 = getH4(temp)
h5_p4 = getH5(temp)

temp = makeDataSet("h3")
h1_p5 = getH1(temp)
h2_p5 = getH2(temp)
h3_p5 = getH3(temp)
h4_p5 = getH4(temp)
h5_p5 = getH5(temp)

#After 5 bags
h1_avg = []
h2_avg = []
h3_avg = []
h4_avg = []
h5_avg = []
for i in range (0,100,1):
    h1_avg.append(statistics.mean([h1_p1[i],h1_p2[i],h1_p3[i],h1_p4[i],h1_p5[i]]))
    h2_avg.append(statistics.mean([h2_p1[i],h2_p2[i],h2_p3[i],h2_p4[i],h2_p5[i]]))
    h3_avg.append(statistics.mean([h3_p1[i],h3_p2[i],h3_p3[i],h3_p4[i],h3_p5[i]]))
    h4_avg.append(statistics.mean([h4_p1[i],h4_p2[i],h4_p3[i],h4_p4[i],h4_p5[i]]))
    h5_avg.append(statistics.mean([h5_p1[i],h5_p2[i],h5_p3[i],h5_p4[i],h5_p5[i]]))
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Bag of h3 after 5 averages")
fig.add_trace(go.Scatter(x = observations, y = h1_avg, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_avg, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_avg, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_avg, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_avg, mode='lines', name='p(h5|d)'))
fig.show()

temp = makeDataSet("h3")
h1_p6 = getH1(temp)
h2_p6 = getH2(temp)
h3_p6 = getH3(temp)
h4_p6 = getH4(temp)
h5_p6 = getH5(temp)

temp = makeDataSet("h3")
h1_p7 = getH1(temp)
h2_p7 = getH2(temp)
h3_p7 = getH3(temp)
h4_p7 = getH4(temp)
h5_p7 = getH5(temp)

temp = makeDataSet("h3")
h1_p8 = getH1(temp)
h2_p8 = getH2(temp)
h3_p8 = getH3(temp)
h4_p8 = getH4(temp)
h5_p8 = getH5(temp)

temp = makeDataSet("h3")
h1_p9 = getH1(temp)
h2_p9 = getH2(temp)
h3_p9 = getH3(temp)
h4_p9 = getH4(temp)
h5_p9 = getH5(temp)

temp = makeDataSet("h3")
h1_p10 = getH1(temp)
h2_p10 = getH2(temp)
h3_p10 = getH3(temp)
h4_p10 = getH4(temp)
h5_p10 = getH5(temp)

#After 10 bags
h1_avg = []
h2_avg = []
h3_avg = []
h4_avg = []
h5_avg = []
for i in range (0,100,1):
    h1_avg.append(statistics.mean([h1_p1[i],h1_p2[i],h1_p3[i],h1_p4[i],h1_p5[i],h1_p6[i],h1_p7[i],h1_p8[i],h1_p9[i],h1_p10[i]]))
    h2_avg.append(statistics.mean([h2_p1[i],h2_p2[i],h2_p3[i],h2_p4[i],h2_p5[i],h2_p6[i],h2_p7[i],h2_p8[i],h2_p9[i],h2_p10[i]]))
    h3_avg.append(statistics.mean([h3_p1[i],h3_p2[i],h3_p3[i],h3_p4[i],h3_p5[i],h3_p6[i],h3_p7[i],h3_p8[i],h3_p9[i],h3_p10[i]]))
    h4_avg.append(statistics.mean([h4_p1[i],h4_p2[i],h4_p3[i],h4_p4[i],h4_p5[i],h4_p6[i],h4_p7[i],h4_p8[i],h4_p9[i],h4_p10[i]]))
    h5_avg.append(statistics.mean([h5_p1[i],h5_p2[i],h5_p3[i],h5_p4[i],h5_p5[i],h5_p6[i],h5_p7[i],h5_p8[i],h5_p9[i],h5_p10[i]]))
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Bag of h3 after 10 averages")
fig.add_trace(go.Scatter(x = observations, y = h1_avg, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_avg, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_avg, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_avg, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_avg, mode='lines', name='p(h5|d)'))
fig.show()


"""===================================================================================================================================================================================================================================================="""
#Getting the Averages for h1

temp = makeDataSet("h1")
h1_p1 = getH1(temp)
h2_p1 = getH2(temp)
h3_p1 = getH3(temp)
h4_p1 = getH4(temp)
h5_p1 = getH5(temp)

#Single Bag
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Single bag of h1 candy")
fig.add_trace(go.Scatter(x = observations, y = h1_p1, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_p1, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_p1, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_p1, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_p1, mode='lines', name='p(h5|d)'))
fig.show()

temp = makeDataSet("h1")
h1_p2 = getH1(temp)
h2_p2 = getH2(temp)
h3_p2 = getH3(temp)
h4_p2 = getH4(temp)
h5_p2 = getH5(temp)

temp = makeDataSet("h1")
h1_p3 = getH1(temp)
h2_p3 = getH2(temp)
h3_p3 = getH3(temp)
h4_p3 = getH4(temp)
h5_p3 = getH5(temp)

temp = makeDataSet("h1")
h1_p4 = getH1(temp)
h2_p4 = getH2(temp)
h3_p4 = getH3(temp)
h4_p4 = getH4(temp)
h5_p4 = getH5(temp)

temp = makeDataSet("h1")
h1_p5 = getH1(temp)
h2_p5 = getH2(temp)
h3_p5 = getH3(temp)
h4_p5 = getH4(temp)
h5_p5 = getH5(temp)

#After 5 bags
h1_avg = []
h2_avg = []
h3_avg = []
h4_avg = []
h5_avg = []
for i in range (0,100,1):
    h1_avg.append(statistics.mean([h1_p1[i],h1_p2[i],h1_p3[i],h1_p4[i],h1_p5[i]]))
    h2_avg.append(statistics.mean([h2_p1[i],h2_p2[i],h2_p3[i],h2_p4[i],h2_p5[i]]))
    h3_avg.append(statistics.mean([h3_p1[i],h3_p2[i],h3_p3[i],h3_p4[i],h3_p5[i]]))
    h4_avg.append(statistics.mean([h4_p1[i],h4_p2[i],h4_p3[i],h4_p4[i],h4_p5[i]]))
    h5_avg.append(statistics.mean([h5_p1[i],h5_p2[i],h5_p3[i],h5_p4[i],h5_p5[i]]))
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Bag of h1 after 5 averages")
fig.add_trace(go.Scatter(x = observations, y = h1_avg, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_avg, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_avg, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_avg, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_avg, mode='lines', name='p(h5|d)'))
fig.show()

temp = makeDataSet("h1")
h1_p6 = getH1(temp)
h2_p6 = getH2(temp)
h3_p6 = getH3(temp)
h4_p6 = getH4(temp)
h5_p6 = getH5(temp)

temp = makeDataSet("h1")
h1_p7 = getH1(temp)
h2_p7 = getH2(temp)
h3_p7 = getH3(temp)
h4_p7 = getH4(temp)
h5_p7 = getH5(temp)

temp = makeDataSet("h1")
h1_p8 = getH1(temp)
h2_p8 = getH2(temp)
h3_p8 = getH3(temp)
h4_p8 = getH4(temp)
h5_p8 = getH5(temp)

temp = makeDataSet("h1")
h1_p9 = getH1(temp)
h2_p9 = getH2(temp)
h3_p9 = getH3(temp)
h4_p9 = getH4(temp)
h5_p9 = getH5(temp)

temp = makeDataSet("h1")
h1_p10 = getH1(temp)
h2_p10 = getH2(temp)
h3_p10 = getH3(temp)
h4_p10 = getH4(temp)
h5_p10 = getH5(temp)

#After 10 bags
h1_avg = []
h2_avg = []
h3_avg = []
h4_avg = []
h5_avg = []
for i in range (0,100,1):
    h1_avg.append(statistics.mean([h1_p1[i],h1_p2[i],h1_p3[i],h1_p4[i],h1_p5[i],h1_p6[i],h1_p7[i],h1_p8[i],h1_p9[i],h1_p10[i]]))
    h2_avg.append(statistics.mean([h2_p1[i],h2_p2[i],h2_p3[i],h2_p4[i],h2_p5[i],h2_p6[i],h2_p7[i],h2_p8[i],h2_p9[i],h2_p10[i]]))
    h3_avg.append(statistics.mean([h3_p1[i],h3_p2[i],h3_p3[i],h3_p4[i],h3_p5[i],h3_p6[i],h3_p7[i],h3_p8[i],h3_p9[i],h3_p10[i]]))
    h4_avg.append(statistics.mean([h4_p1[i],h4_p2[i],h4_p3[i],h4_p4[i],h4_p5[i],h4_p6[i],h4_p7[i],h4_p8[i],h4_p9[i],h4_p10[i]]))
    h5_avg.append(statistics.mean([h5_p1[i],h5_p2[i],h5_p3[i],h5_p4[i],h5_p5[i],h5_p6[i],h5_p7[i],h5_p8[i],h5_p9[i],h5_p10[i]]))
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Bag of h1 after 10 averages")
fig.add_trace(go.Scatter(x = observations, y = h1_avg, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_avg, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_avg, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_avg, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_avg, mode='lines', name='p(h5|d)'))
fig.show()


"""===================================================================================================================================================================================================================================================="""
#Getting the Averages for h2

temp = makeDataSet("h2")
h1_p1 = getH1(temp)
h2_p1 = getH2(temp)
h3_p1 = getH3(temp)
h4_p1 = getH4(temp)
h5_p1 = getH5(temp)

#Single Bag
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Single bag of h2 candy")
fig.add_trace(go.Scatter(x = observations, y = h1_p1, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_p1, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_p1, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_p1, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_p1, mode='lines', name='p(h5|d)'))
fig.show()

temp = makeDataSet("h2")
h1_p2 = getH1(temp)
h2_p2 = getH2(temp)
h3_p2 = getH3(temp)
h4_p2 = getH4(temp)
h5_p2 = getH5(temp)

temp = makeDataSet("h2")
h1_p3 = getH1(temp)
h2_p3 = getH2(temp)
h3_p3 = getH3(temp)
h4_p3 = getH4(temp)
h5_p3 = getH5(temp)

temp = makeDataSet("h2")
h1_p4 = getH1(temp)
h2_p4 = getH2(temp)
h3_p4 = getH3(temp)
h4_p4 = getH4(temp)
h5_p4 = getH5(temp)

temp = makeDataSet("h2")
h1_p5 = getH1(temp)
h2_p5 = getH2(temp)
h3_p5 = getH3(temp)
h4_p5 = getH4(temp)
h5_p5 = getH5(temp)

#After 5 bags
h1_avg = []
h2_avg = []
h3_avg = []
h4_avg = []
h5_avg = []
for i in range (0,100,1):
    h1_avg.append(statistics.mean([h1_p1[i],h1_p2[i],h1_p3[i],h1_p4[i],h1_p5[i]]))
    h2_avg.append(statistics.mean([h2_p1[i],h2_p2[i],h2_p3[i],h2_p4[i],h2_p5[i]]))
    h3_avg.append(statistics.mean([h3_p1[i],h3_p2[i],h3_p3[i],h3_p4[i],h3_p5[i]]))
    h4_avg.append(statistics.mean([h4_p1[i],h4_p2[i],h4_p3[i],h4_p4[i],h4_p5[i]]))
    h5_avg.append(statistics.mean([h5_p1[i],h5_p2[i],h5_p3[i],h5_p4[i],h5_p5[i]]))
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Bag of h2 after 5 averages")
fig.add_trace(go.Scatter(x = observations, y = h1_avg, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_avg, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_avg, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_avg, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_avg, mode='lines', name='p(h5|d)'))
fig.show()

temp = makeDataSet("h2")
h1_p6 = getH1(temp)
h2_p6 = getH2(temp)
h3_p6 = getH3(temp)
h4_p6 = getH4(temp)
h5_p6 = getH5(temp)

temp = makeDataSet("h2")
h1_p7 = getH1(temp)
h2_p7 = getH2(temp)
h3_p7 = getH3(temp)
h4_p7 = getH4(temp)
h5_p7 = getH5(temp)

temp = makeDataSet("h2")
h1_p8 = getH1(temp)
h2_p8 = getH2(temp)
h3_p8 = getH3(temp)
h4_p8 = getH4(temp)
h5_p8 = getH5(temp)

temp = makeDataSet("h2")
h1_p9 = getH1(temp)
h2_p9 = getH2(temp)
h3_p9 = getH3(temp)
h4_p9 = getH4(temp)
h5_p9 = getH5(temp)

temp = makeDataSet("h2")
h1_p10 = getH1(temp)
h2_p10 = getH2(temp)
h3_p10 = getH3(temp)
h4_p10 = getH4(temp)
h5_p10 = getH5(temp)

#After 10 bags
h1_avg = []
h2_avg = []
h3_avg = []
h4_avg = []
h5_avg = []
for i in range (0,100,1):
    h1_avg.append(statistics.mean([h1_p1[i],h1_p2[i],h1_p3[i],h1_p4[i],h1_p5[i],h1_p6[i],h1_p7[i],h1_p8[i],h1_p9[i],h1_p10[i]]))
    h2_avg.append(statistics.mean([h2_p1[i],h2_p2[i],h2_p3[i],h2_p4[i],h2_p5[i],h2_p6[i],h2_p7[i],h2_p8[i],h2_p9[i],h2_p10[i]]))
    h3_avg.append(statistics.mean([h3_p1[i],h3_p2[i],h3_p3[i],h3_p4[i],h3_p5[i],h3_p6[i],h3_p7[i],h3_p8[i],h3_p9[i],h3_p10[i]]))
    h4_avg.append(statistics.mean([h4_p1[i],h4_p2[i],h4_p3[i],h4_p4[i],h4_p5[i],h4_p6[i],h4_p7[i],h4_p8[i],h4_p9[i],h4_p10[i]]))
    h5_avg.append(statistics.mean([h5_p1[i],h5_p2[i],h5_p3[i],h5_p4[i],h5_p5[i],h5_p6[i],h5_p7[i],h5_p8[i],h5_p9[i],h5_p10[i]]))
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Bag of h2 after 10 averages")
fig.add_trace(go.Scatter(x = observations, y = h1_avg, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_avg, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_avg, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_avg, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_avg, mode='lines', name='p(h5|d)'))
fig.show()

"""===================================================================================================================================================================================================================================================="""
#Getting the Averages for h4

temp = makeDataSet("h4")
h1_p1 = getH1(temp)
h2_p1 = getH2(temp)
h3_p1 = getH3(temp)
h4_p1 = getH4(temp)
h5_p1 = getH5(temp)

#Single Bag
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Single bag of h4 candy")
fig.add_trace(go.Scatter(x = observations, y = h1_p1, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_p1, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_p1, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_p1, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_p1, mode='lines', name='p(h5|d)'))
fig.show()

temp = makeDataSet("h4")
h1_p2 = getH1(temp)
h2_p2 = getH2(temp)
h3_p2 = getH3(temp)
h4_p2 = getH4(temp)
h5_p2 = getH5(temp)

temp = makeDataSet("h4")
h1_p3 = getH1(temp)
h2_p3 = getH2(temp)
h3_p3 = getH3(temp)
h4_p3 = getH4(temp)
h5_p3 = getH5(temp)

temp = makeDataSet("h4")
h1_p4 = getH1(temp)
h2_p4 = getH2(temp)
h3_p4 = getH3(temp)
h4_p4 = getH4(temp)
h5_p4 = getH5(temp)

temp = makeDataSet("h4")
h1_p5 = getH1(temp)
h2_p5 = getH2(temp)
h3_p5 = getH3(temp)
h4_p5 = getH4(temp)
h5_p5 = getH5(temp)

#After 5 bags
h1_avg = []
h2_avg = []
h3_avg = []
h4_avg = []
h5_avg = []
for i in range (0,100,1):
    h1_avg.append(statistics.mean([h1_p1[i],h1_p2[i],h1_p3[i],h1_p4[i],h1_p5[i]]))
    h2_avg.append(statistics.mean([h2_p1[i],h2_p2[i],h2_p3[i],h2_p4[i],h2_p5[i]]))
    h3_avg.append(statistics.mean([h3_p1[i],h3_p2[i],h3_p3[i],h3_p4[i],h3_p5[i]]))
    h4_avg.append(statistics.mean([h4_p1[i],h4_p2[i],h4_p3[i],h4_p4[i],h4_p5[i]]))
    h5_avg.append(statistics.mean([h5_p1[i],h5_p2[i],h5_p3[i],h5_p4[i],h5_p5[i]]))
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Bag of h4 after 5 averages")
fig.add_trace(go.Scatter(x = observations, y = h1_avg, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_avg, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_avg, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_avg, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_avg, mode='lines', name='p(h5|d)'))
fig.show()

temp = makeDataSet("h4")
h1_p6 = getH1(temp)
h2_p6 = getH2(temp)
h3_p6 = getH3(temp)
h4_p6 = getH4(temp)
h5_p6 = getH5(temp)

temp = makeDataSet("h4")
h1_p7 = getH1(temp)
h2_p7 = getH2(temp)
h3_p7 = getH3(temp)
h4_p7 = getH4(temp)
h5_p7 = getH5(temp)

temp = makeDataSet("h4")
h1_p8 = getH1(temp)
h2_p8 = getH2(temp)
h3_p8 = getH3(temp)
h4_p8 = getH4(temp)
h5_p8 = getH5(temp)

temp = makeDataSet("h4")
h1_p9 = getH1(temp)
h2_p9 = getH2(temp)
h3_p9 = getH3(temp)
h4_p9 = getH4(temp)
h5_p9 = getH5(temp)

temp = makeDataSet("h4")
h1_p10 = getH1(temp)
h2_p10 = getH2(temp)
h3_p10 = getH3(temp)
h4_p10 = getH4(temp)
h5_p10 = getH5(temp)

#After 10 bags
h1_avg = []
h2_avg = []
h3_avg = []
h4_avg = []
h5_avg = []
for i in range (0,100,1):
    h1_avg.append(statistics.mean([h1_p1[i],h1_p2[i],h1_p3[i],h1_p4[i],h1_p5[i],h1_p6[i],h1_p7[i],h1_p8[i],h1_p9[i],h1_p10[i]]))
    h2_avg.append(statistics.mean([h2_p1[i],h2_p2[i],h2_p3[i],h2_p4[i],h2_p5[i],h2_p6[i],h2_p7[i],h2_p8[i],h2_p9[i],h2_p10[i]]))
    h3_avg.append(statistics.mean([h3_p1[i],h3_p2[i],h3_p3[i],h3_p4[i],h3_p5[i],h3_p6[i],h3_p7[i],h3_p8[i],h3_p9[i],h3_p10[i]]))
    h4_avg.append(statistics.mean([h4_p1[i],h4_p2[i],h4_p3[i],h4_p4[i],h4_p5[i],h4_p6[i],h4_p7[i],h4_p8[i],h4_p9[i],h4_p10[i]]))
    h5_avg.append(statistics.mean([h5_p1[i],h5_p2[i],h5_p3[i],h5_p4[i],h5_p5[i],h5_p6[i],h5_p7[i],h5_p8[i],h5_p9[i],h5_p10[i]]))
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Bag of h4 after 10 averages")
fig.add_trace(go.Scatter(x = observations, y = h1_avg, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_avg, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_avg, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_avg, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_avg, mode='lines', name='p(h5|d)'))
fig.show()


"""===================================================================================================================================================================================================================================================="""
#Getting the Averages for h5

temp = makeDataSet("h5")
h1_p1 = getH1(temp)
h2_p1 = getH2(temp)
h3_p1 = getH3(temp)
h4_p1 = getH4(temp)
h5_p1 = getH5(temp)

#Single Bag
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Single bag of h5 candy")
fig.add_trace(go.Scatter(x = observations, y = h1_p1, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_p1, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_p1, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_p1, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_p1, mode='lines', name='p(h5|d)'))
fig.show()

temp = makeDataSet("h5")
h1_p2 = getH1(temp)
h2_p2 = getH2(temp)
h3_p2 = getH3(temp)
h4_p2 = getH4(temp)
h5_p2 = getH5(temp)

temp = makeDataSet("h5")
h1_p3 = getH1(temp)
h2_p3 = getH2(temp)
h3_p3 = getH3(temp)
h4_p3 = getH4(temp)
h5_p3 = getH5(temp)

temp = makeDataSet("h5")
h1_p4 = getH1(temp)
h2_p4 = getH2(temp)
h3_p4 = getH3(temp)
h4_p4 = getH4(temp)
h5_p4 = getH5(temp)

temp = makeDataSet("h5")
h1_p5 = getH1(temp)
h2_p5 = getH2(temp)
h3_p5 = getH3(temp)
h4_p5 = getH4(temp)
h5_p5 = getH5(temp)

#After 5 bags
h1_avg = []
h2_avg = []
h3_avg = []
h4_avg = []
h5_avg = []
for i in range (0,100,1):
    h1_avg.append(statistics.mean([h1_p1[i],h1_p2[i],h1_p3[i],h1_p4[i],h1_p5[i]]))
    h2_avg.append(statistics.mean([h2_p1[i],h2_p2[i],h2_p3[i],h2_p4[i],h2_p5[i]]))
    h3_avg.append(statistics.mean([h3_p1[i],h3_p2[i],h3_p3[i],h3_p4[i],h3_p5[i]]))
    h4_avg.append(statistics.mean([h4_p1[i],h4_p2[i],h4_p3[i],h4_p4[i],h4_p5[i]]))
    h5_avg.append(statistics.mean([h5_p1[i],h5_p2[i],h5_p3[i],h5_p4[i],h5_p5[i]]))
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Bag of h5 after 5 averages")
fig.add_trace(go.Scatter(x = observations, y = h1_avg, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_avg, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_avg, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_avg, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_avg, mode='lines', name='p(h5|d)'))
fig.show()

temp = makeDataSet("h5")
h1_p6 = getH1(temp)
h2_p6 = getH2(temp)
h3_p6 = getH3(temp)
h4_p6 = getH4(temp)
h5_p6 = getH5(temp)

temp = makeDataSet("h3")
h1_p7 = getH1(temp)
h2_p7 = getH2(temp)
h3_p7 = getH3(temp)
h4_p7 = getH4(temp)
h5_p7 = getH5(temp)

temp = makeDataSet("h5")
h1_p8 = getH1(temp)
h2_p8 = getH2(temp)
h3_p8 = getH3(temp)
h4_p8 = getH4(temp)
h5_p8 = getH5(temp)

temp = makeDataSet("h5")
h1_p9 = getH1(temp)
h2_p9 = getH2(temp)
h3_p9 = getH3(temp)
h4_p9 = getH4(temp)
h5_p9 = getH5(temp)

temp = makeDataSet("h5")
h1_p10 = getH1(temp)
h2_p10 = getH2(temp)
h3_p10 = getH3(temp)
h4_p10 = getH4(temp)
h5_p10 = getH5(temp)

#After 10 bags
h1_avg = []
h2_avg = []
h3_avg = []
h4_avg = []
h5_avg = []
for i in range (0,100,1):
    h1_avg.append(statistics.mean([h1_p1[i],h1_p2[i],h1_p3[i],h1_p4[i],h1_p5[i],h1_p6[i],h1_p7[i],h1_p8[i],h1_p9[i],h1_p10[i]]))
    h2_avg.append(statistics.mean([h2_p1[i],h2_p2[i],h2_p3[i],h2_p4[i],h2_p5[i],h2_p6[i],h2_p7[i],h2_p8[i],h2_p9[i],h2_p10[i]]))
    h3_avg.append(statistics.mean([h3_p1[i],h3_p2[i],h3_p3[i],h3_p4[i],h3_p5[i],h3_p6[i],h3_p7[i],h3_p8[i],h3_p9[i],h3_p10[i]]))
    h4_avg.append(statistics.mean([h4_p1[i],h4_p2[i],h4_p3[i],h4_p4[i],h4_p5[i],h4_p6[i],h4_p7[i],h4_p8[i],h4_p9[i],h4_p10[i]]))
    h5_avg.append(statistics.mean([h5_p1[i],h5_p2[i],h5_p3[i],h5_p4[i],h5_p5[i],h5_p6[i],h5_p7[i],h5_p8[i],h5_p9[i],h5_p10[i]]))
fig = go.Figure()
fig.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Bag of h5 after 10 averages")
fig.add_trace(go.Scatter(x = observations, y = h1_avg, mode='lines', name='p(h1|d)'))
fig.add_trace(go.Scatter(x = observations, y = h2_avg, mode='lines', name='p(h2|d)'))
fig.add_trace(go.Scatter(x = observations, y = h3_avg, mode='lines', name='p(h3|d)'))
fig.add_trace(go.Scatter(x = observations, y = h4_avg, mode='lines', name='p(h4|d)'))
fig.add_trace(go.Scatter(x = observations, y = h5_avg, mode='lines', name='p(h5|d)'))
fig.show()



"""===================================================================================================================================================================================================================================================="""
















































