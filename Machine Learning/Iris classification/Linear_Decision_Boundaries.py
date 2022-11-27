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
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

pio.renderers.default = 'browser'

#This module contain the source code for Question 2

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

#This function will plot the second and third iris classes
def plot_iris(df):
    fig = px.scatter(df,x="petal_length",y="petal_width",color="species",title="Iris classes")
    fig.show()



#%%

#This method will plot the output from the neural network over the given values from the dataset
def plot_Results(df,_class):
    #Plotting the scatterplot using plotly
    fig = px.scatter(df, x="petal_length",y="petal_width",color=_class,title="Output from Linear Classification using sigmoid non-linearity")
    fig.show()


#%%

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
    fig_1.update_layout(title="Linear Decision boundary")
    fig_1.show()



#%%

#This function will plot the output of the neural network over all input space
def plot_over_space(weights, bias):
    #Creating a dataframe of all of the input and output points 
    z = []
    #x = np.linspace(0.0,10.0,100)
    #y = np.linspace(0.0,10.0,100)
    x = []
    y = []
    for i in range(0,100,1):
        for j in range(0,100,1):
            x.append(i)
            y.append(j)
            arr = np.array([i,j])
            temp = np.dot(arr, weights) + bias
            temp = sigmoid(temp)
            z.append(temp)
    
    #z = np.array(z)
    temp_df = pd.DataFrame(list(zip(x, y, z)),columns =['x', 'y','z'])
    
    x1 = np.linspace(temp_df['x'].min(), temp_df['x'].max(), len(temp_df['x'].unique()))
    y1 = np.linspace(temp_df['y'].min(), temp_df['y'].max(), len(temp_df['y'].unique()))

    x2, y2 = np.meshgrid(x1, y1)

    z2 = griddata((temp_df['x'], temp_df['y']), temp_df['z'], (x2, y2), method='cubic')

    # Ready to plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_zlim(0, 1.01)

    #Setting for the plot
    fig.colorbar(surf, aspect=5)
    plt.title('Output of neural network over all input space')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')

    plt.show()
   


#%%
#Main method of the module
def main():
    #print("Hello World!")
    
    #Getting the dataframe from the csv
    df = pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")
    
    #Extracting the 2nd and 3rd classes from the data frame
    df = df[df["species"] != "setosa"]

    #Plotting the second or third iris classes
    plot_iris(df)
    
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
    
    #End of data preprocessing
    
    #Weight vector and bias scalars
    w = np.array([-0.05,0.51])
    b = -0.6
    
    #Predicting based on input weights and bias
    _class = predict(vals,w,b)
    
    #error = misclassifiction_count(species, _class)
    #print("The error with the current weights and biases is: ",error)
    
    #Calling the method that will plot the classification of the neural network
    plot_Results(df, _class)
    
    #Calling the function that will plot the linear decision boundary
    get_Linear_db(df, w, b, _class)
    
    #Calling the function that will plot the results of the neural network over all in
    plot_over_space(w, b)
    
    
    
    
    
    
    
    
    
#%%

if __name__ == "__main__":
    main()
    
#%%



