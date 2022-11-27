"""
Created on Tue Nov 15 13:37:15 2022

@author: rohansingh
"""

#Module for 2 Implementation of KMeans clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
import random
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'

#this is used to get the iris data
df = pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")

#Class for 2-D implementation of KMeans algortihm
class KMeans_2D:
    
    #Constructor
    def __init__(self, k=3, max_iter=100,df=pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")):
        self.k = k
        self.max_iter=max_iter
        self.df = df
        
    #Helper method to find the 2D Euclidean distance
    def euc_dist(self,point, centroid):
        dist_sum = 0
        dist_sum += np.power((point[0]-centroid[0]),2)
        dist_sum += np.power((point[1]-centroid[1]),2)
        return np.sqrt(dist_sum)
    
    #Pre-processing the data for convenience
    def getPoints(self):
        #Retrieving the relevant columns from the dataframe
        p_len = self.df["petal_length"]
        p_wid = self.df["petal_width"]
        
        #Converting the columns into a numpy array
        p_len = p_len.to_numpy()
        p_wid = p_wid.to_numpy()
        
        #Adding the daat points into an array of Coordinates (Each coordinate itself is an array of 2 elements)
        points = []
        for i in range(0,len(p_len),1):
            #Each row is a points of petal length and petal width
            row = []
            row.append(p_len[i])
            row.append(p_wid[i])
            points.append(row)
            
        return points
    
    #Helper function for KMeans Objective Function
    def objectiveFunction(self,classes,centroids,points):
        #Iterating through all of the points
        val = 0
        
        for i in range(0,len(points),1):
            cl = -1
            dist = 1000000
            
            #Finding the closest centroid
            for j in range(0,len(centroids),1):
                #Getting the euclidean distance between the 2 points
                temp = self.euc_dist(points[i],centroids[j])
                
                #If the new distance is the shortest, then we update 
                if(temp < dist):
                    dist = temp
                    cl = j
                    
            #Appending the point based on the class that it belongs to 
            classes[cl].append(points[i])
            val += dist
        return classes
    
    
    #Helper function for the update rule for KMeans
    def updateCentroids(self,centroids, classes):
        #updating the coordinates of the centroids for each class
        new_centroids = []
        
        #The new centroids become the mean of each cluster
        for i in range(0,self.k,1):
            tempclass = classes[i]
            p_len_sum = 0
            p_wid_sum = 0
            
            #The new value of the attributes of the centroids is the mean of the attribute in the class
            for j in range(0,len(tempclass),1):
                point = tempclass[j]
                p_len_sum += point[0]
                p_wid_sum += point[1]
            cl = []
            cl.append(p_len_sum/len(tempclass))
            cl.append(p_wid_sum/len(tempclass))
            new_centroids.append(cl)
            
        return new_centroids
    
    def predict(self):
        
        #Getting thhe points
        points = self.getPoints()
        centroids = []
        classes = []
        
        #Setting the centroids to a random 'k' points
        centroids = random.sample(points,k=self.k)
        
        #Adding the centroids to their respective classes
        for i in range(0,self.k,1):
            ls = []
            ls.append(centroids[i])
            classes.append(ls)
            
        #Reupdating the centoids until the max limit
        for i in range(0,self.max_iter,1):
            #Getting the clusters
            classes = self.objectiveFunction(classes,centroids,points)
            
            #Updating the value of the centroids using the update function
            new_centroids = self.updateCentroids(centroids,classes)
            
            #Checking if we have reached the optimal position, that is, the mean doesn't change that much
            opt = True
            
            for j in range(0,self.k,1):
                a = centroids[j]
                b = new_centroids[j]
                for k in range(0,2,1):
                    c = a[k]
                    d = b[k]
                    #If we are not in the optimal position
                    if(abs((c-d)/d) > 0.0001):
                        opt = False
            
            if(opt):
                break
            
            centroids = new_centroids
        
        #Converting the data (points and class) into a pandas dataframe
        p_len = []
        p_wid = []
        class_ = []
        for i in range(0,len(classes),1):
            temp_class = classes[i]
            for j in range(0,len(temp_class),1):
                class_.append(i)
                a = temp_class[j]
                p_len.append(a[0])
                p_wid.append(a[1])

        df_1 = pd.DataFrame(list(zip(p_len,p_wid,class_)),columns=["Petal Length","Petal Width","Class"])
        
        return df_1
    
    #Method that predicts and plots
    def predict_and_plot(self,_title):
        #Plotting the DataFrame from the predict function using plotly
        df_1 = self.predict()
        fig = px.scatter(df_1, x="Petal Length", y="Petal Width", color="Class")
        fig.update_layout(title=_title)
        fig.show()
    

def main():
        Temp = KMeans_2D(k=3)
        Temp.predict_and_plot("KMeans clustering for k = 3")

        Temp = KMeans_2D(k=2)
        Temp.predict_and_plot("KMeans clustering for k = 2")
    
if __name__ == '__main__':
    main()
    
    
    
    
    