# -*- coding: utf-8 -*-
"""
21 May 2022
@author: Rohan Singh
"""

#%%

print("Hello World!\n")




#%%

class Graph:
    
    #initializing the graph
    def __init__(self):
        self.Matrix = []        #List of vertices will be represented by "Matrix"
        #self.tempPQ = []
        #self.finalized = []
    
    #function to add a vertex to the graph
    def addVertex(self, name, value):
        #Checking to see if the graph already has a vertex with the same name
        if (self.getIndex(name) != -1):
            return
        #Appending a new vertex to the list of Vertices
        self.Matrix.append(Vertex(name, value))
        
    """   
    Printing the contents of the graph in the following format:
    sourceVertex -> destinationvertex1 destinationVertex2 destinationVertex3
    """
    def printGraph(self):
        for i in range (0, len(self.Matrix)):        #Going to each node of the graph
            str = ""
            str += self.Matrix[i].name + " -> "
            for j in range (0, len(self.Matrix[i].myList)):      #Going through the list of neighbors
                toIndex = self.Matrix[i].myList[j].index
                str += self.Matrix[toIndex].name + " "
            print(str + "\n")
            
    def addEdge(self, fromVertex, toVertex, cost):
        #Finding the indices of the vertices provided
        fromIndex = self.getIndex(fromVertex)
        toIndex = self.getIndex(toVertex)
        #Leaving if the vertex doesn't exist
        if((fromIndex==-1)or(toIndex==-1)):
            return
        self.Matrix[fromIndex].addEdge(toIndex,cost)
    
    
    
    
    
#%%

class Vertex:
    
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.myList = []
        self.parents = []
        
    def addEdge(self,index,cost):
        self.myList.append(Edge(index,cost))
        
    
#%%

class Edge:
    
    def __init__(self,index,cost):
        self.index = index
        self.cost = cost
        
#%%