"""
Created on Tue May 10 13:50:58 2022
@author: rohansingh
"""



#%%
class UCS_Graph:
    
    #initializing the graph
    def __init__(self):
        self.Matrix = []        #List of vertices will be represented by "Matrix"
        self.tempPQ = []
        self.finalized = []
    
    #function to add a vertex to the graph
    def addVertex(self, name):
        #Checking to see if the graph already has a vertex with the same name
        if (self.getIndex(name) != -1):
            return
        #Appending a new vertex to the list of Vertices
        self.Matrix.append(Vertex(name))
        
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
     
    #Function to add an edge to the Graph
    def addEdge(self, fromVertex, toVertex, cost):
        #Finding the indices of the vertices provided
        fromIndex = self.getIndex(fromVertex)
        toIndex = self.getIndex(toVertex)
        #Leaving if the vertex doesn't exist
        if((fromIndex==-1)or(toIndex==-1)):
            return
        self.Matrix[fromIndex].addEdge(toIndex,cost)
        
    #Function for Uniform Cost Search
    def Uniform_Cost_Search(self,fromVertex):
        #The list to store costs
        dj = []
        
        #The list to store the finalized vertices
        self.finalized = []
        self.tempQ = []
        numFinalized = 0
        
        #Starting index
        startIndex = self.getIndex(fromVertex)
        
        #initialization
        for i in range (0, len(self.Matrix)):
            self.finalized.append(False)
            dj.append(9223372036854775807)
            self.Matrix[i].parents.append(startIndex)
            
        #Initializing the PQ, by adding the source vertex
        self.tempPQ.append(Edge(startIndex,0))
        #Cost to the source vertex from the source vertex is 0
        dj[startIndex] = 0
        
        #Iterating while the priority queue isn't empty and the number of finalized vertices is less than the number of elements
        while ((len(self.tempPQ) > 0) and (numFinalized < len(self.Matrix))):
            #Getting the lowest cost entry
            e = self.findSmallest(self.tempPQ)
            currIndex = e.index
            
            #If the node was previously been finalized, then we will continue
            if(self.finalized[currIndex] == True):
                continue
               
            #Finalizing the lowest cost node
            self.finalized[currIndex] = True
            numFinalized += 1
            
            #Traversing through the neighbors of the lowest cost vertex/node
            for i in range (0, len(self.Matrix[currIndex].myList)):
                tempIndex = self.Matrix[currIndex].myList[i].index
                
                #If the neighbor hasn't been finalized
                if(self.finalized[tempIndex] == False):
                    #Calculating the new cost
                    tempCost = dj[currIndex] + self.Matrix[currIndex].myList[i].cost
                    
                    #If the new cost is lower then we will change the entries
                    if(tempCost < dj[tempIndex]):
                        dj[tempIndex] = tempCost
                        self.Matrix[tempIndex].parents.append(currIndex)
                        self.tempPQ.append(Edge(tempIndex,tempCost))
            
        #Printing out the results using helper methods
        print("From the Vertex " + fromVertex + ":\n")
        self.printDijkstrasDistance(dj)
        print("\n\n")
        self.printDijkstrasPath()
        
    #Helper method to print out the path
    def printDijkstrasPath(self):
        for i in range(0,len(self.Matrix)):
            tempu = self.Matrix[i].parents
            str1 = ""
            for j in range (0,len(tempu)):
                str1 += (self.Matrix[tempu[j]].name + " -> ")
            printStr = self.Matrix[i].name + ": Path is " + str1 + self.Matrix[i].name + "\n"
            print(printStr)
    
    #Helper method to print out the distance
    def printDijkstrasDistance(self, arr):
        for i in range (0, len(self.Matrix)):
            printStr = self.Matrix[i].name + "\t\t\t" + str(arr[i]) + "\n"
            print(printStr)
      
            
      
    #Helper method to find the lowest edge cost  
    def findSmallest(self, arr):
        if(len(arr) == 0):
            return
        smallest = arr[0]
        smallestIndex = 0
        #Iterating through the array to find the lowest cost variable
        for i in range (0, len(arr)):
            #Updating the cost and index if a new minimum is found
            if(arr[i].cost < smallest.cost):
                smallest = arr[i]
                smallestIndex = i
        tempu = arr[smallestIndex]
        del arr[smallestIndex]
        return tempu
    
    #Helper method to find the index of the vertex with the given name
    def getIndex(self, vertexName):
        #Iterating through the matrix to find the vertex with the same name
        for i in range (0, len(self.Matrix)):
            if(vertexName == self.Matrix[i].name):
                return i
        return -1
        
            
        


#%%

#Class for Vertex
class Vertex:
    
    #Contains attributes for: name, list of neighbors and list of parents
    
    def __init__(self,name):
        self.name = name
        self.myList = []
        self.parents = []
        
    def addEdge(self,index,cost):
        self.myList.append(Edge(index,cost))
        
        


#%%

#Class for Edge
class Edge:
    
    #Contains attributes for: index of edge and cost
    
    def __init__(self,index,cost):
        self.index = index
        self.cost = cost
 
    
 
#%%

#Main method to demonstrate the Uniform Cost Search Algorithm
def main():
    g = UCS_Graph()
    
    g.addVertex("A")
    g.addVertex("B")
    g.addVertex("C")
    g.addVertex("D")
    g.addVertex("E")
    g.addVertex("F")
    g.addVertex("G")
    
    g.addEdge("A","B",5)
    g.addEdge("A","C",3)
    g.addEdge("B","C",2)
    g.addEdge("B","E",3)
    g.addEdge("B","G",1)
    g.addEdge("C","D",7)
    g.addEdge("C","E",7)
    g.addEdge("D","A",1)
    g.addEdge("D","F",6)
    g.addEdge("E","D",2)
    g.addEdge("E","F",1)
    g.addEdge("G","E",1)
    
    g.printGraph()
    
    print("\n\n")
    
    g.dijkstrasAlgo("A")
    
    
    
    
#%%

if __name__ == "__main__":
    main()
    
    
    
#%%
