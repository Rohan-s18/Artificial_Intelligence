"""
Created on Fri Nov 25 14:07:24 2022

@author: rohansingh
"""

#Python module that calculates the mean squared error for a given set of parameters in a neural network
#This module contains source code for Question 3 (a) and (b)

#Imports
import pandas as pd
import numpy as np
import plotly.express as px
import random
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#%%
"""
This block of code contains the code for the neural network
"""

#Function that calculates the value of the sigmoid function
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

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

"""
End of neural network code
"""
#%%

"""
This is the main function that we will be using
"""

#Function to calculate the mean squareed error
def calculate_MSE(data,weights,bias,target):
    
    #Predicting using the neural network predict function
    y = predict(data,weights,bias)
    mse = 0
    
    #Iterating through all of the data points
    for i in range(0,len(y),1):
        #Adding the sum of the differences
        mse += ((y[i] - target[i])**2)
    
    #Dividing the sum by 2
    mse /= 2
    
    return mse

#%%
#This method will plot the linear decision boundary
def plot_Linear_db(df,data,w,b,_title):
    #Plotting the decision boundary

    #Getting the inctercept and the slope
    c = -(b/w[1])
    m = -(w[0]/w[1])
    
    #Making the arrays for the x and y axes
    xd = np.array([2.5,7])
    yd = m*xd + c
    
    #Getting the classes
    _class = predict(data,w,b)
    
    fig_1 = go.Figure()
    fig_1.add_trace(
                go.Scatter(x=df["petal_length"], y=df["petal_width"],
                mode='markers',
                name='points',
                marker = {'color':_class}
                ))
    fig_1.add_trace(go.Scatter(x=xd, y=yd,
                mode='lines',
                name='decision boundary'))
    fig_1.update_layout(title=_title)
    fig_1.show()

#%%

#The main method will be used to calculate the mean squared error
def main():
    #print("Hello World!")
    #Getting the dataframe from the csv
    df = pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")

    #Extracting the 2nd and 3rd classes from the data frame
    df = df[df["species"] != "setosa"]
    vals = df.iloc[:,2:4]
    target = df.iloc[:,-1]

    vals = vals.to_numpy()
    target = target.to_numpy()

    #Changing specie names to binary values 
    temp = []
    for i in range (0,len(target),1):
        if(target[i] == "versicolor"):
            temp.append(0)
        else:
            temp.append(1)
    target = np.array(temp)
    
    #These are the optimum weights and bias:
    optimum_w = np.array([-0.05,0.51])
    optimum_b = -0.6
    optimum_mse = calculate_MSE(vals, optimum_w, optimum_b, target)
    
    #These are some random weights and bias:
    rand_w = np.array([56.89,-34.15])
    rand_b = 23.8
    rand_mse = calculate_MSE(vals, rand_w, rand_b, target)
    
    #These are close to optimal weights and bias:
    close_w = np.array([-0.1,0.55])
    close_b = -0.5
    close_mse = calculate_MSE(vals, close_w, close_b, target)
    
    print("Optimal MSE: ",optimum_mse)
    print("Random MSE: ",rand_mse)
    print("Close MSE: ",close_mse)
    
    #Plotting the linear decision boundary for small error parameters
    plot_Linear_db(df, vals, close_w, close_b,"Linear Decision Boundary for small error")
    
    #Plotting the linear decision boundary for large error parameters
    plot_Linear_db(df, vals, rand_w, rand_b,"Linear Decision Boundary for large error")
    
    
    
#%%

if __name__ == "__main__":
    main()
    
#%%
