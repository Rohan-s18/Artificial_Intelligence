#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 18:16:37 2022

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

#Class that we will use for Sigmoid Decision Boundary
class Sigmoid_Descision_Boundary:
    
    def __init__(self,epoch,learning_rate):
        self.epochs = epoch
        self.lr = learning_rate
        self.coef_ = None
        self.intercept_ = None
        
    def fit (self,X_train,y_train):
        self.coef_ = np.zeros(X_train.shape[1])
        self.intercept_ = 0
        
        for i in range(self.epochs):
            y_hat = self.sigmoid(np.dot(X_train,self.coef_) + self.intercept_)
            intercept_def = -np.mean(y_train - y_hat)
            coef_der = -np.dot((y_train - y_hat),X_train)
            self.coef_ = self.coef_ - self.lr*coef_der
            self.intercept_ = self.intercept_ - self.lr*intercept_def
        return self.coef_,self.intercept_
        
    #Helper method to calculate the sigmoid function value
    def sigmoid(self, a):
        return (1 / (1 + np.exp(-a)))

    #Helper function to calculate the likelihood
    def likelihood():
        print("Hello World!")
        
    #Prediction function
    def predict(self):
        print("Hello World!")

#Main function of the module
def main():
    
        #Retrieving the dataframe from the input csv
        df = pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")
        df = df[df["species"] != "setosa"]
        
        #Plotting the data using plottly
        # fig = px.scatter(df, x="petal_length", y="petal_width", color="species")
        # fig.update_layout(title="Known Versicolor and Virginica")
        # fig.show()
        
        #Getting the petal length and width from the dataframe
        X = df.iloc[:,2:4]
        y = df.iloc[:,-1]
        
        Tester = Sigmoid_Descision_Boundary(500, 0.01)
        coef,intercept = Tester.fit(X, y)
        
        #Getting the eqn of line constants
        m = -(coef[0]/coef[1])
        b = -(intercept/coef[1])
        x_input = np.linspace(4,7,100)
        y_input = m*x_input + b
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X["petal_length"], y=X["petal_width"],
                    mode='markers',
                    name='points'))
        fig.add_trace(go.Scatter(x=x_input, y=y_input,
                    mode='lines',
                    name='prediction'))
        fig.show()
        
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    