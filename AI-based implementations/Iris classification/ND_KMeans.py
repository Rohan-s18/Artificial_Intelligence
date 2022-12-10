"""
Created on Thu Nov  3 10:20:04 2022

@author: rohansingh
"""

# Source code for K-means clustering

#Getting imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
import random
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'

#this is used to get the iris data
df = pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")

#Code for KMeans class
class KMeans:
    
    def __init__(self, k=3, max_iter=100,df=pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")):
        self.k = k
        self.max_iter=max_iter
        self.df = df
        
    #Helper function to get the euclidean distance
    def euc_dist(self,point, centroid):
        dist_sum = 0
        dist_sum += np.power((point[0]-centroid[0]),2)
        dist_sum += np.power((point[1]-centroid[1]),2)
        dist_sum += np.power((point[2]-centroid[2]),2)
        dist_sum += np.power((point[3]-centroid[3]),2)
        return np.sqrt(dist_sum)
    
    #Pre-processing the data for convenience
    def getPoints(self):
        s_len = self.df["sepal_length"]
        s_wid = self.df["sepal_width"]
        p_len = self.df["petal_length"]
        p_wid = self.df["petal_width"]
        
        s_len = s_len.to_numpy()
        s_wid = s_wid.to_numpy()
        p_len = p_len.to_numpy()
        p_wid = p_wid.to_numpy()
        
        points = []
        species = []
        for i in range(0,len(s_len),1):
            row = []
            row.append(s_len[i])
            row.append(s_wid[i])
            row.append(p_len[i])
            row.append(p_wid[i])
            points.append(row)
            species.append(-1)
            
        return points,species
    
    #Helper function for KMeans Objective Function
    def objectiveFunction(self,classes,centroids,points,species):
        #This will hold the sum of the objective function value
        val = 0            

        #Iterating through all of the points
        for i in range(0,len(points),1):
            cl = -1
            dist = 1000000
            #Finding the closest centroid for the j-th point
            for j in range(0,len(centroids),1):
                temp = self.euc_dist(points[i],centroids[j])
                if(temp < dist):
                    dist = temp
                    cl = j
            classes[cl].append(points[i])
            species[i] = cl
            #Adding the closest distance to the value
            val += dist
        return classes,species,val
    
    
    #Helper function for the update rule for KMeans
    def updateCentroids(self,centroids, classes):
        #updating the coordinates of the centroids for each class
        new_centroids = []

        #This is done by taking the mean of each cluster
        for i in range(0,self.k,1):
            tempclass = classes[i]
            s_len_sum = 0
            s_wid_sum = 0
            p_len_sum = 0
            p_wid_sum = 0
            for j in range(0,len(tempclass),1):
                point = tempclass[j]
                s_len_sum += point[0]
                s_wid_sum += point[1]
                p_len_sum += point[2]
                p_wid_sum += point[3]
            cl = []
            cl.append(s_len_sum/len(tempclass))
            cl.append(s_wid_sum/len(tempclass))
            cl.append(p_len_sum/len(tempclass))
            cl.append(p_wid_sum/len(tempclass))
            new_centroids.append(cl)
            
        return new_centroids
    
    #Function to predict/classify the points into clusters
    def predict(self):
        
        points,species = self.getPoints()
        centroids = []
        classes = []
        
        #Setting the centroids to a random 'k' points
        centroids = random.sample(points,k=self.k)

        for i in range(0,self.k,1):
            ls = []
            ls.append(points[i])
            classes.append(ls)
            
        
        objective_function_vals = []
        for i in range(0,self.max_iter,1):
            #Getting the clusters
            classes,species,func_val = self.objectiveFunction(classes,centroids,points,species)
            
            objective_function_vals.append(func_val)

            #Updating the value of the centroids using the update function
            new_centroids = self.updateCentroids(centroids,classes)
            
            #Checking if we have reached the optimal position
            opt = True
            
            for j in range(0,self.k,1):
                a = centroids[j]
                b = new_centroids[j]
                for k in range(0,4,1):
                    c = a[k]
                    d = b[k]
                    if(abs((c-d)/d) > 0.0001):
                        opt = False
            
            if(opt):
                break
            
            centroids = new_centroids
            
        
        return classes,objective_function_vals

    #This function will plot the minimization fo the objective function
    def plot_obj_function(self,vals):
        x_axis = []
        for i in range(0,len(vals),1):
            x_axis.append(i)

        #Converting it into a dataframe
        temp_df = pd.DataFrame(list(zip(x_axis,vals)),columns=["Iteration","Objective Function Value"])

        #Plotting the graph
        fig = px.line(temp_df, x="Iteration", y="Objective Function Value", title="Reduction in objective function value")
        fig.show()


#%%
def main():
    print("Hello World!")
    Temp = KMeans(k=3)

    #Getting the classes for ND_Kmeans (a)
    _classes, vals = Temp.predict()

    #Plotting the reduction in the objective function (b)
    Temp.plot_obj_function(vals)








#%%
if __name__ == "__main__":
    main()

#%%
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
