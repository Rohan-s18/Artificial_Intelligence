#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:20:04 2022

@author: rohansingh
"""

# Source code for K-means clustering

#Getting imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

#this is used to get the iris data
df = pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")

def plotIris():
    X = df.data[:, 3:]
    y = df.target
    mtp.scatter(X[:, 0],X[:, 1],c=y)
    mtp.xlabel("Petal Length")
    mtp.ylabel("Petal Width")
    mtp.show()
  
def eucDist(x1, x2, y1, y2):
    return np.sqrt( (np.power((x2-x1),2)) + (np.power((y2-y1),2)) )

def objFunction(mu, pos1, pos2):
    return 0



class KMeans:
    
    #Constructor for the KMeans class
    def __init__(self, k=3, max_iter=100,df=pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")):
        self.k = k
        self.max_iter=max_iter
        self.df = df
        
    #Helper function to find the euclidean distance between a point and a centroid
    def euc_dist(self,point, centroid):
        #Getting the square of the differences between all of the attributes 
        dist_sum = 0
        dist_sum += np.power((point[0]-centroid[0]),2)
        dist_sum += np.power((point[1]-centroid[1]),2)
        dist_sum += np.power((point[2]-centroid[2]),2)
        dist_sum += np.power((point[3]-centroid[3]),2)
        
        #Returning the squareroot of the sum of square distances
        return np.sqrt(dist_sum)
    
    #Pre-processing the data for convenience
    def getPoints(self):
        #Getting the individual pandas series from the dataframe
        s_len = self.df["sepal_length"]
        s_wid = self.df["sepal_width"]
        p_len = self.df["petal_length"]
        p_wid = self.df["petal_width"]
        
        #Converting the pandas series to numpy arrays
        s_len = s_len.to_numpy()
        s_wid = s_wid.to_numpy()
        p_len = p_len.to_numpy()
        p_wid = p_wid.to_numpy()
        
        #Points are arrays that hold individual point vectors
        points = []
        species = []
        for i in range(0,len(s_len),1):
            #Adding the data from the arrays to the row elements
            row = []
            row.append(s_len[i])
            row.append(s_wid[i])
            row.append(p_len[i])
            row.append(p_wid[i])
            points.append(row)
            species.append(-1)
           
        #Returning the points and an empty list of species
        return points,species
    
    #Helper function for KMeans Objective Function
    def objectiveFunction(self,classes,centroids,points,species):
        #Iterating through all of the points
        val = 0
        for i in range(0,len(points),1):
            cl = -1
            dist = 1000000
            
            #iterating through all of the centroids to find the closest one
            for j in range(0,len(centroids),1):
                temp = self.euc_dist(points[i],centroids[j])
                
                #Reupdating the minimum distance and the closest centroid if the distance is less than the minimum distance
                if(temp < dist):
                    dist = temp
                    cl = j
            
            #Adding the point to the class it belongs to
            classes[cl].append(points[i])
            species[i] = cl
            val += dist
            
        #Returning the classes and species lists
        return classes,species
    
    
    #Helper function for the update rule for KMeans
    def updateCentroids(self,centroids, classes):
        #updating the coordinates of the centroids for each class to make it the average
        new_centroids = []
        
        #Iterating through all of the classes
        for i in range(0,self.k,1):
            tempclass = classes[i]
            s_len_sum = 0
            s_wid_sum = 0
            p_len_sum = 0
            p_wid_sum = 0
            
            #Iterating through all of the points of eahc class
            for j in range(0,len(tempclass),1):
                point = tempclass[j]
                s_len_sum += point[0]
                s_wid_sum += point[1]
                p_len_sum += point[2]
                p_wid_sum += point[3]
                
            #Creating a vector to store the coordinates of the new centroid
            cl = []
            #Appending the averages of each attribute to the vector
            cl.append(s_len_sum/len(tempclass))
            cl.append(s_wid_sum/len(tempclass))
            cl.append(p_len_sum/len(tempclass))
            cl.append(p_wid_sum/len(tempclass))
            
            #Adding the average vector to the new_centroids list
            new_centroids.append(cl)
        
        #Returning the new centroids
        return new_centroids
    
    #Creating the predict function for the KMeans clustering algorithm
    def predict(self):
        
        #Getting the points
        points,species = self.getPoints()
        centroids = []
        classes = []
        
        #Setting the centroids to a random 'k' points
        centroids = random.sample(points,k=self.k)
        
        #Adding the centroids to their respective classes
        for i in range(0,self.k,1):
            ls = []
            ls.append(centroids[i])
            classes.append(ls)
            
        
        for i in range(0,self.max_iter,1):
            #Getting the clusters
            classes,species = self.objectiveFunction(classes,centroids,points,species)
            
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
            
        
        return classes
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
