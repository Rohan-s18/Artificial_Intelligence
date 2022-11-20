#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 19:12:53 2022

@author: rohansingh
"""
#Imports
import pandas as pd
import numpy as np
import plotly.express as px
import random
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = 'browser'

#%%
#Function that calculates the value of the sigmoid function
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

#%%
#Function that predicts the value of the class using a single-layer neural network
def predict(X,weights,bias):
    # X --> Input vector
    
    # Calculating the value of y using the inner-product (dot product) of the weights vecotr with the data vector and adding a bias term
    y = np.dot(X, weights) + bias
    preds = sigmoid(y)
    
    # Empty List to store predictions.
    pred_class = []
    for i in range(0,len(preds),1):
        # if preds >= 0.5 --> round up to 1
        # if preds < 0.5 --> round up to 1
        if(preds[i] > 0.5):
            pred_class.append(1)
        else:
            pred_class.append(0)
    
    #pred_class = [1 if i > 0.5 else 0 for i in preds]
    
    return np.array(pred_class)

#%%
#This method will count the number of misclassifications
def misclassifiction_count(test, predicted_vals):
    #Counter variable
    ct = 0
    
    #Iterating through each entry
    for i in range(0,len(test),1):
        #This will add the number of misclassifications
        ct += abs(test[i] - predicted_vals[i])
        
    return ct


#%%
#Main method of the module
def main():
    #print("Hello World!")
    
    #Getting the dataframe from the csv
    df = pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")
    
    #Extracting the 2nd and 3rd classes from the data frame
    df = df[df["species"] != "setosa"]
    
    #Seperating the values for petal length/width from the species
    vals = df.iloc[:,2:4]
    species = df.iloc[:,-1]
    
    vals = vals.to_numpy()
    species = species.to_numpy()
    
    #Changing species from names to binary values 
    temp = []
    for i in range (0,len(species),1):
        if(species[i] == "versicolor"):
            temp.append(0)
        else:
            temp.append(1)
    species = np.array(temp)
    
    #Weight vector and bias scalars
    w = np.array([-0.05,0.51])
    b = -0.6
    
    #Predicting based on input weights and bias
    _class = predict(vals,w,b)
    
    error = misclassifiction_count(species, _class)
    print("The error with the current weights and biases is: ",error)
    
    #Plotting the scatterplot using plotly
    fig = px.scatter(df, x="petal_length",y="petal_width",color=_class)
    fig.show()
    
    
    
    
#%%

if __name__ == "__main__":
    main()
    
#%%







