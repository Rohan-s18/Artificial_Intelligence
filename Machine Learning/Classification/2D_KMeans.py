#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:37:15 2022

@author: rohansingh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

#this is used to get the iris data
df = pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")

class KMeans_2D:
    
    def __init__(self, k=3, max_iter=100,df=pd.read_csv("/Users/rohansingh/Documents/CSDS 391/AI_Code/Machine Learning/Classification/irisdata.csv")):
        self.k = k
        self.max_iter=max_iter
        self.df = df
        
    def euc_dist(self,point, centroid):
        dist_sum = 0
        dist_sum += np.power((point[0]-centroid[0]),2)
        dist_sum += np.power((point[1]-centroid[1]),2)
        return np.sqrt(dist_sum)
    
    #Pre-processing the data for convenience
    def getPoints(self):
        p_len = self.df["petal_length"]
        p_wid = self.df["petal_width"]
        
        p_len = p_len.to_numpy()
        p_wid = p_wid.to_numpy()
        
        points = []
        for i in range(0,len(p_len),1):
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
            for j in range(0,len(centroids),1):
                temp = self.euc_dist(points[i],centroids[j])
                if(temp < dist):
                    dist = temp
                    cl = j
            classes[cl].append(points[i])
            val += dist
        return classes
    
    
    #Helper function for the update rule for KMeans
    def updateCentroids(self,centroids, classes):
        #updating the coordinates of the centroids for each class
        new_centroids = []
        for i in range(0,self.k,1):
            tempclass = classes[i]
            p_len_sum = 0
            p_wid_sum = 0
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
        
        points = self.getPoints()
        centroids = []
        classes = []
        
        for i in range(0,self.k,1):
            centroids.append(points[i])
            ls = []
            ls.append(points[i])
            classes.append(ls)
            
        
        for i in range(0,self.max_iter,1):
            #Getting the clusters
            classes = self.objectiveFunction(classes,centroids,points)
            
            #Updating the value of the centroids using the update function
            new_centroids = self.updateCentroids(centroids,classes)
            
            #Checking if we have reached the optimal position
            opt = True
            
            for j in range(0,self.k,1):
                a = centroids[j]
                b = new_centroids[j]
                for k in range(0,2,1):
                    c = a[k]
                    d = b[k]
                    if(abs((c-d)/d) > 0.0001):
                        opt = False
            
            if(opt):
                break
            
            centroids = new_centroids
            
        
        return classes
    
    
    
    
    
    
    
    
    