"""
Author: Nishita Singh
Python Module for testing hypothesis 1 and 2
"""


#  Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#%%
"""
This cell contains the code for the Linear Regression based model
"""

# Class for multivariate linear regression
class LinReg:


    #  Initialization function
    def __init__(self, dataset, targets, n):
        self.trainset = dataset
        self.targetset = targets
        self.n = n


    #  Function to calculate f(x) values 
    def get_output(self, weights):

        # This is the vector to store the function output
        output = []

        for i in range(0, len(self.trainset), 1):

            a = np.array([1])
            a = np.concatenate(a, self.trainset[i],axis=0)

            #Calculating the function value for the i-th point
            output.append(np.dot(a,weights))

        return np.array(output)
    

    #  This helper function will be used to calculate the total squared error for the neural network
    def get_TSE(self, weights):

        y = self.get_output(weights)

        tse = 0.0

        #Iterating through all of the training points and calculating the total squared error
        for i in range(0, len(y), 1):

            tse += ((y[i] - self.targetset[i])**2)

        
        #Dividing the sum by 2
        tse /= 2

        return tse
    

    #  This function will calculate the summed gradient that will be used in the learning rule
    def get_summed_gradient(self, weights):
        
        #  This will hold the gradien tfo rall of the n parameters
        gradient = []

        y = self.get_output(weights)

        bias_sum = 0

        for j in range(0,len(self.trainset), 1):

            temp = y[j] - self.targetset[j]

            bias_sum += temp

        gradient.append(bias_sum)

        for i in range(0, self.n, 1):

            grad_sum = 0

            for j in  range(0,len(self.trainset),1):

                temp = y[j] - self.targetset[j]

                #Multiplying the difference with the datapoint's i-th component
                temp *= self.trainset[j][i]

                grad_sum += temp

            #Appending the gradient sum for the i-th feature to the gradient vector
            grad_sum /= len(self.trainset)
            gradient.append(grad_sum)

        return np.array(gradient)
    

    #This will be used to train the model using gradient descent, using the input step-size and stopping point and a maximum iteration count
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
    
    #This will be used as the predict function for the model after the training on the test-set
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
    
#%%

"""
This cell contains a helper function to pre-process the data for the demonstration in the main method
"""

def get_data_1(filepath):
    dataset = []
    target = []

    #Reading the csv from the filepath
    df = pd.read_csv(filepath)

    #Getting numpy array columns from the dataframe
    fO = df["family ownership"].to_numpy()
    #FP = df["Firm performance"].to_numpy()
    ITI = df["IT investment"].to_numpy()

    for i in range(0,len(fO),1):

        #For a row of datapoints
        temp = []
        temp.append(fO[i])
        #temp.append(FP[i])
        
        target.append(ITI[i])


        #Adding the row to the dataset
        dataset.append(temp)


    return np.array(dataset), np.array(target)


def get_data_2(filepath):
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

    for i in range(0,len(fO),1):

        #For a row of datapoints
        temp = []
        temp.append(fO[i])
        temp.append(EH[i])
        temp.append(PE[i])
        
        target.append(FP[i])
        
        #target.append(ITI[i])


        #Adding the row to the dataset
        dataset.append(temp)


    return np.array(dataset), np.array(target)

def get_data_3(filepath):
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

    for i in range(0,len(fO),1):

        #For a row of datapoints
        temp = []
        temp.append(fO[i])
        temp.append(EH[i])
        temp.append(PE[i])
        
        target.append(FP[i])
        
        #target.append(ITI[i])


        #Adding the row to the dataset
        dataset.append(temp)


    return np.array(dataset), np.array(target)

    


#%%

"""
This function contains the code for running the main function
"""

def main():

    hyp_1_filepath = ""
    hyp_2_filepath = ""
    
    

    #Doing the analysis for Hypothesis 1
    fO, ITI = get_data_1(hyp_1_filepath)

    Hyp1_model = LinReg(dataset=fO, targets=ITI, n=1)
    Hyp1_model.train(0.001, 500, 10000)
    out = LinReg.predict(fO)


    #Doing the analysis for Hypothesis 2
    data , FP = get_data_2(hyp_2_filepath)
    Hyp2_model = LinReg(dataset=data, targets=FP, n=3)
    Hyp2_model.train(0.001, 500, 10000)
    out = LinReg.predict(data)



if __name__ == "__main__":
    main()
    








#%%
