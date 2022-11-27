"""
Created on Fri Nov 25 20:17:46 2022

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

#This module contains the source code for Question 3 (e)

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


#This method will plot the linear decision boundary
def get_Linear_db(df,w,b,_class):
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
    fig_1.show()


"""
End of neural network code
"""
#%%
"""
This section includes the code that is required for Q3 (e)
"""


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

#%%
#This contains the main method
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
    
    #Getting the change in the linear decision boundary using the function 
    change_in_db(df,np.array([0.0,0.0]),0,vals,target,0.0001,100)
    

#%%
if __name__ == "__main__":
    main()
    
#%%









