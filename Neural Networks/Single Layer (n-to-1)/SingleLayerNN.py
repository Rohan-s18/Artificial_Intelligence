# Author: Rohan Singh
# Date: 12/9/2022
# Source code for a general purpose single-layer n-to-1 neural network, i.e. The network takes in an n-dimensional input vector and returns a scalar output

#imports 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
"""
This cell contains the source code for the SingleLayer Neural Network Class
"""
class NeuralNetwork:

    #The constructor for the Neural Network class
    def __init__(self, dataset, targets, n):
        self.trainset = dataset
        self.targetset = targets
        self.n = n
        pass

    

    #This function will give you the output of the neural network given a weight vector
    def get_NN_Output(self, weights):

        #This vector will contain the output of the neural network
        output = []

        #Getting the output for all of the datapoints within the dataset
        for i in range (0,len(self.trainset),1):

            #This will be added for the bias-term
            a = np.array([1]) 
            a = np.concatenate((a,self.trainset[i]),axis=0)

            #Appending the output for the point (inner-product) into the output list
            output.append(np.dot(a,weights))

        #Converting the output into a numpy array
        return np.array(output)


    #This function will find the total squared error for the Neural Network from a given set of weights 
    def get_TSE(self,weights):

        #Predicting thee output of the neural network for the given weights
        y = self.get_NN_Output(weights=weights)

        #This variable will store the total squared error for the given output
        tse = 0.0

        #Iterating through all of the data points
        for i in range(0,len(y),1):
            #Adding the squared differences
            tse += ((y[i] - self.targetset[i])**2)
        
        #Dividing the sum by 2
        tse /=2 

        return tse

    #This function will calculate the summed gradient that will be used in the learning rule
    def get_summed_gradient(self,weights):
        
        #This list will hold the summed gradient for all of the 'n' parameters
        gradient = []

        #Getting the temporary predictions for the given set of weights
        y = self.get_NN_Output(weights=weights)

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

        

        return np.array(gradient)


    #This will be used to train the Neural Network using gradient descent, using the input step-size and stopping point and a maximum iteration count
    def train(self, epsilon, maxerr, maxiter):
        
        #Setting the intial weights to 0 for all of the n-parameters (and the bias)
        w = np.zeros(self.n + 1)

        #Iterating within the maximum iteration limit
        for i in range (0,maxiter,1):

            #Getting the predicted values given the current weights
            y = self.get_NN_Output(w)

            #Geting the total squared error for the temporary output
            tse = self.get_TSE(w)

            #Checking if the mean-squared error has minimized to the threshold
            mse = tse/(len(self.trainset))
            if(mse < maxerr):
                break

            #Getting the summed gradient
            summed_grad = self.get_summed_gradient(w)
            
            #Updating the weights for the next iteration
            w -= (epsilon*summed_grad)

        self.trained_weight = w
        #Returning the optimal weights 
        return w

    #This will be used as the predict function for the neural-network after the training on the test-set
    def predict(self, test_set):
        
        #This vector will contain the output of the neural network
        output = []

        #Getting the output for all of the datapoints within the test set
        for i in range (0,len(test_set),1):

            #This will be added for the bias-term
            a = np.array([1]) 
            a = np.concatenate((a,test_set[i]),axis=0)

            #Appending the output for the point (inner-product with the optimum weights) into the output list
            output.append(np.dot(a,self.trained_weight))

        #Converting the output into a numpy array
        return np.array(output)

    def printHello(self):
        print("Hello World!")


#%%
"""
This cell contains a helper function to pre-process the data for the demonstration in the main method
"""

def get_data():
    dataset = []
    target = []

    filepath = "/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv"

    #Reading the csv from the filepath
    df = pd.read_csv(filepath)

    #Getting numpy array columns from the dataframe
    pet_len = df["petal_length"].to_numpy()
    pet_wid = df["petal_width"].to_numpy()
    sep_len = df["sepal_length"].to_numpy()
    sep_wid = df["sepal_width"].to_numpy()
    spec = df["species"]

    for i in range(0,len(pet_len),1):

        #For a row of datapoints
        temp = []
        temp.append(pet_len[i])
        temp.append(pet_wid[i])
        temp.append(sep_len[i])
        temp.append(sep_wid[i])

        #Converting Species string to float
        if(spec[i] == "setosa"):
            target.append(100.0)
        elif(spec[i] == "virginica"):
            target.append(200.0)
        else:
            target.append(300.0)

        #Adding the row to the dataset
        dataset.append(temp)


    return np.array(dataset), np.array(target)


#%%
"""
This cell contains the main method to show the demonstration of the code
"""

def main():

    #Fetching the data using the helper function in the cell above
    dataset, target = get_data()

    #Instantiating the Neural Network Object
    DemoNN = NeuralNetwork(dataset=dataset,targets=target,n=4)
    #out = DemoNN.get_NN_Output(np.array([1,1,1,1,1]))
    DemoNN.train(0.001,500,10000)
    out = DemoNN.predict(dataset)
    print(out)



if __name__ == "__main__":
    main()


#%%
