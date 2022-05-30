#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:12:19 2022

@author: rohansingh
"""


#%%
import matplotlib.pyplot as plt
import numpy as np

class plots:
    
    def xyline():
        xAxis = np.array([0,10])
        yAxis = np.array([0,100])
    
        plt.plot(xAxis,yAxis)
        plt.show()
    



    def pointsPlot():
        xPoints = np.array([1,3,5,7,9])
        yPoints = np.array([4,8,6,10,12])
        
        plt.plot(xPoints, yPoints, 'o')
        plt.show()
        
    def zigzagLine():
        xPoints = np.array([0,2,8,12])
        yPoints = np.array([10,2,8,8])
        
        plt.plot(xPoints,yPoints)
        plt.show()

    def defaultX():
        yPoints = np.array([10,-5,-2,-3,2,4,6,4,0])
        
        plt.plot(yPoints)
        plt.show()
        
    #Format String syntax- marker|line|color
    def formattedPlot():
        ypoints = np.array([3,8,1,10])
        
        formatStr = "X-.c"
        plt.plot(ypoints, formatStr, ms = 10, mec = 'r', mfc = 'b')
        plt.show()
        
    def multipleLines():
        x1 = np.array([1,3,5,7])
        x2 = np.array([0,2,4,6,8])
        x3 = np.array([0,1,2,3,4,5])
        y1 = np.array([1,3,5,7])
        y2 = np.array([10,8,2,6,4])
        y3 = np.array([4,6,5,6,6,7])
        
        plt.plot(x1,y1,x2,y2,x3,y3)
        plt.show()
        
    def label():
        yPoints = np.array([0,2,4,6,8,10])
        
        plt.plot(yPoints)
        plt.title("Displacement-Time Graph")
        plt.xlabel("Time")
        plt.ylabel("Displacement")
        plt.show()
        

#%%