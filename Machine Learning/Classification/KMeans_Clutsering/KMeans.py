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
    
    def __init__(self, k=3, max_iter=100,df=pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")):
        self.k = k
        self.max_iter=max_iter
        self.df = df
        
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
        #Iterating through all of the points
        val = 0
        for i in range(0,len(points),1):
            cl = -1
            dist = 1000000
            for j in range(0,len(centroids),1):
                temp = self.euc_dist(points[i],centroids[j])
                if(temp < dist):
                    dist = temp
                    cl = j
            classes[cl].append(points[i])
            species[i] = cl
            val += dist
        return classes,species
    
    
    #Helper function for the update rule for KMeans
    def updateCentroids(self,centroids, classes):
        #updating the coordinates of the centroids for each class
        new_centroids = []
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
    
    def predict(self):
        
        points,species = self.getPoints()
        centroids = []
        classes = []
        
        for i in range(0,self.k,1):
            centroids.append(points[i])
            ls = []
            ls.append(points[i])
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
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
