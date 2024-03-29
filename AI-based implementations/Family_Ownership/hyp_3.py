# Author: Nishita Singh
# Python Module for testing hypothesis 3

#imports 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
"""
This cell contains the source code for the mulativariate linear regression model
"""
class LinReg_3:

    #The constructor for the model class
    def __init__(self, dataset, targets, n, m):
        self.trainset = dataset
        self.targetset = targets
        self.n = n
        self.m = m
        self.init_w = np.zeros([m,n+1])
        pass

    

    #This function will give you the output of the model given a weight vector
    def get_output(self, weights):

        #This vector will contain the output of the model
        output = []

        #Getting the output for all of the datapoints within the dataset
        for i in range (0,len(self.trainset),1):

            #This will be added for the bias-term
            a = np.array([1]) 
            a = np.concatenate((a,self.trainset[i]),axis=0)

            #Appending the output for the point (inner-product) into the output list
            output.append(np.matmul(weights,a))

        #Converting the output into a numpy array
        return np.array(output)


    #This function will find the total squared error for the model from a given set of weights 
    def get_TSE(self,weights):

        #Predicting thee output of the model for the given weights
        y = self.get_output(weights=weights)

        #This variable will store the total squared error for the given output
        tse = 0.0

        #Iterating through all of the data points
        for i in range(0,len(y),1):
            #Adding the squared differences
            tse += ((y[i] - self.targetset[i])**2)
        
        #Dividing the sum by 2
        tse /=2 

        sum = 0
        for i in range(0,len(tse),1):
            sum += tse[i]

        return sum

    #This function will calculate the summed gradient that will be used in the learning rule
    def get_summed_gradient(self,weights):
        
        #This list will hold the summed gradient for all of the 'n' parameters
        gradient = []

        #Getting the temporary predictions for the given set of weights
        y = self.get_output(weights=weights)

        bias_sum = 0
        #Iterating through all of the data/target points
        for j in range(0,len(self.trainset),1):
            #Finding the difference between the prediction and the target
            temp = y[j] - self.targetset[j]

            #Adding the difference to the bias sum
            bias_sum += temp

        gradient.append(bias_sum)

        #Iterating through all of the 'n' parameter dimensions
        for i in range(0,self.n,1):

            #Calculating the gradient sum for the parameter
            grad_sum = 0
            
            #Iterating through all of the data/target points
            for j in range(0,len(self.trainset),1):
                #Finding the difference between the prediction and the target
                temp = y[j] - self.targetset[j]

                #Multiplying the difference with the datapoint's i-th component
                temp *= self.trainset[j][i]

                grad_sum += temp

            #Appending the gradient sum for the i-th feature to the gradient vector
            grad_sum /= len(self.trainset)
            gradient.append(grad_sum)

        
        gradient = np.array(gradient)
        return np.transpose(gradient)

    #This function will be used to create a matrix of initial weights
    def initial_weights(self):
        
        #The list of rows represents the matrix
        rows = []
        for i in range(0,self.m,1):

            #This represents an individual row
            row = []
            for j in range(0,self.n+1,1):
                row.append(0.0)

            rows.append(row)

        return np.array(rows)

    #This will be used to train the model using gradient descent, using the input step-size and stopping point and a maximum iteration count
    def train(self, epsilon, maxerr, maxiter):
        
        #Setting the intial weights to 0 for all of the n-parameters (and the bias), this makes the initial matrix an MxN matrix
        w = self.initial_weights()

        #Iterating within the maximum iteration limit
        for i in range (0,maxiter,1):

            #Getting the predicted values given the current weights
            y = self.get_output(w)

            #Geting the total squared error for the temporary output
            tse = self.get_TSE(w)

            #Checking if the mean-squared error has minimized to the threshold
            mse = tse/(len(self.trainset))
            if(mse < maxerr):
                break

            #Getting the summed gradient
            summed_grad = self.get_summed_gradient(w)
            
            #Updating the weights for the next iteration
            summed_grad *= epsilon
            w -= summed_grad

        self.trained_weight = w
        #Returning the optimal weights 
        return w

    #This will be used as the predict function for the model after the training on the test-set
    def predict(self, test_set):
        
        #This vector will contain the output of the model
        output = []

        #Getting the output for all of the datapoints within the test set
        for i in range (0,len(test_set),1):

            #This will be added for the bias-term
            a = np.array([1]) 
            a = np.concatenate((a,test_set[i]),axis=0)

            #Appending the output for the point (inner-product with the optimum weights) into the output list
            output.append(np.matmul(self.trained_weight,a))

        #Converting the output into a numpy array
        return np.array(output)

    def printHello(self):
        print("Hello World!")


#%%
"""
This cell contains a helper function to pre-process the data for the demonstration in the main method
"""

def get_data(filepath):
    dataset = []
    target = []

    #Reading the csv from the filepath
    df = pd.read_csv(filepath)

    #Getting numpy array columns from the dataframe
    fO = df["family ownership"].to_numpy()
    FP = df["Firm performance"].to_numpy()

    EH = df["environment hostility"].to_numpy()
    PE = df["professional executive"].to_numpy()

    #ITI = df["IT investment"].to_numpy()

    for i in range(0,len(FP),1):

        #For a row of datapoints
        data_row = []
        data_row.append(fO[i])
        data_row.append(FP[i])

        #For a row of the targets
        target_row = []
        target_row.append(EH[i])
        target_row.append(PE[i])

        #Adding the row to the dataset
        dataset.append(data_row)

        #Adding  the row to the target
        target.append(target_row)


    return np.array(dataset), np.array(target)


#%%
"""
This cell contains the main method to show the demonstration of the code
"""

def main():

    #Fetching the data using the helper function in the cell above
    dataset, target = get_data()

    #Instantiating the model Object
    Hyp3_model = LinReg_3(dataset=dataset,targets=target,n=2,m=2)

    #Training the model
    out = Hyp3_model.train(0.0001,5,10000)

    #Using the predict function of the model
    out = Hyp3_model.predict(dataset)
    print(out)



if __name__ == "__main__":
    main()


#%%