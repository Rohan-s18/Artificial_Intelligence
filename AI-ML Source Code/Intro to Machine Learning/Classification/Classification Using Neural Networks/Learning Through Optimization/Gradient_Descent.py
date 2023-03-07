#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 23:22:14 2022

@author: rohansingh
"""

#Imports
import pandas as pd
import numpy as np
import plotly.express as px
import random
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt

pio.renderers.default = 'browser'

#%%
"""
This block of code contains the helper functions for the neural network
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


#This method will plot the linear decision boundary
def get_Linear_db(df,w,b,_class, plot_title):
    #Plotting the decision boundary
    
    #Getting the inctercept and the slope
    c = -(b/w[1])
    m = -(w[0]/w[1])
    
    #Making the arrays for the x and y axes
    xd = np.array([2.5,7])
    yd = m*xd + c
    
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
    fig_1.update_layout(title = plot_title, xaxis_title = "Petal Length", yaxis_title = "Petal Width")
    fig_1.show()
    
#Function that calculates the summed gradient
def get_summed_gradient(w,b,data,target):
    gradient = []
    bias_gradient = 0
    
    _class = predict(data,w,b)
    
    #For the bias gradient we just have to sum up all of the differences in the prediction and target
    for i in range(0,len(data),1): 
        bias_gradient += _class[i] - target[i]
    
    #Iterating through all of the weights
    for i in range(0,len(w),1):
        grad_sum = 0
        
        #Iterating through all of the data points
        for j in range(0,len(data),1):
            temp = _class[j] - target[j]
            temp *= data[j][i]
            grad_sum += temp
        
        gradient.append(grad_sum)
    
    return np.array(gradient),bias_gradient


#Function to see the change in the decision boundary
def change_in_db(df,init_w,init_b,data,target,step_size,num_iter):
    #Initializing the weights and bias
    w = init_w
    b = init_b
    
    for i in range(0,num_iter,1):
        #Getting the predicted class
        pred = predict(data,w,b)
        
        #Calling the helper function to plot the linear decision boundary
        get_Linear_db(df,w,b,pred)
        
        #Getting the gradients
        weight_grad, bias_grad = get_summed_gradient(w,b,data,target)
        
        #Updating the weights and bias
        w -= step_size*weight_grad
        b -= step_size*bias_grad
  

#Function to calculate the mean squareed error
def calculate_MSE(pred,target):
    #Initializing the value of mse 
    mse = 0
    
    #Iterating through all of the data points
    for i in range(0,len(pred),1):
        #Adding the sum of the differences
        mse += ((pred[i] - target[i])**2)
    
    #Dividing the sum by 2
    mse /= 2
    
    return mse


"""
End of neural network helper functions
"""
#%%

"""
Code for Gradient Descent is over here
"""
#Function for gradient descent
def gradeient_descent(init_w ,init_b ,data ,target ,eps, max_iter):
    #Initializing the weights and bias
    w = init_w
    b = init_b
    error_list = []
    
    w_middle = np.array([0.0,0.0])
    b_middle = 0
    
    w_list = []
    b_list = []
    
    for i in range(0,max_iter,1):
        #Getting the predicted class
        pred = predict(data,w,b)
        
        w_list.append(w)
        b_list.append(b)
        
        #Getting the mean sqaured error
        temp_mse = calculate_MSE(pred, target)
        error_list.append(temp_mse)
        
        #Leaving if the mean sqaured error is less than 10
        if(temp_mse < 5):
            break
        
        #Getting the gradients
        weight_grad, bias_grad = get_summed_gradient(w,b,data,target)
        
        #Updating the weights and bias
        w -= eps*weight_grad
        b -= eps*bias_grad        
        
        
    return w, b, w_list[int(len(w_list)/2)], b_list[int(len(b_list)/2)], error_list


#Function for plotting the error function
def plot_learningcurve(error_list):
    #Getting the x axis
    x = []
    for i in range(0,len(error_list),1):
        x.append(i)
        
    #Creating a pandas dataframe
    temp_df = pd.DataFrame(list(zip(x, error_list)), columns =['x', 'y'])  
    fig = px.line(temp_df, x="x", y="y", title="Error Reduction") 
    fig.show()
    
#Function to get random weights and bias
def get_random_parameters(l_w, u_w, l_b,u_b):
    w = np.array([random.uniform(l_w, u_w),random.uniform(l_w, u_w)])
    b = random.uniform(l_b,u_b)
    return w,b
    

#%%
"""
Main method for demonstration of the code
"""
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
 
    #Changing species from names to binary values 
    temp = []
    for i in range (0,len(target),1):
        if(target[i] == "versicolor"):
            temp.append(0)
        else:
            temp.append(1)
    target = np.array(temp) 
    
    #Getting the initial weights and bias terms using the random generator helper function
    init_w, init_b = get_random_parameters(-1, 1, -1, 1)
    
    #Using the gradient descent to get the optimal weights and bias (as well as error list) as well as the intermediate parameter
    weights, bias, intermediate_weighs, intermediate_bias, error_list = gradeient_descent(init_w, init_b, vals, target, 0.01, 1000)
    
    #Plotting the learning curve
    plot_learningcurve(error_list)
    
    #Plotting the linear decision boundary for the initial weights and bias
    _class = predict(vals, init_w, init_b)
    get_Linear_db(df, init_w, init_b, _class,"Initial Paramters")    
    
    #Plotting the linear decision boundary for the intermediate weights and bias
    _class = predict(vals, intermediate_weighs, intermediate_bias)
    get_Linear_db(df, intermediate_weighs, intermediate_bias, _class,"Intermediate Parameters")
    
    #Plotting the linear decision boundary of the optimum weights and bias
    _class = predict(vals, weights, bias)
    get_Linear_db(df, weights, bias, _class,"Optimum Parameters")
    
    
    
    print(weights)
    print(bias)
    
#%%

if __name__ == "__main__":
    main()

#%%































