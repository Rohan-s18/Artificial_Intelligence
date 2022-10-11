import java.util.*;;

/* Author: Rohan Singh
 * Extra Credit
 * Route Finder in a Graph which uses the A* algorithm and uses Euclidean distance between 2 points as the heuristic.
 * This map is a directional weighted graph
 * */


public class RouteFinder {
	
	 /* That data is represented by multiple Arrays (name, position and the edges)
	 * */
	
	
	//Array of Coordinates of the vertices of the Graph
    private Coordinate[] position;
    
    //Array of the names of the Vertices of the Graph
    private String[] names;
    
    //2D Matrix represents the connections in the Graph
    private double[][] matrix;
    
    //Size counter for the Graph
    private int size;
    private int numVertices;

    //Class representing the x,y coordinates of the vertices
    private class Coordinate{
        double x;
        double y;
        private Coordinate(double x, double y){
            this.x = x;
            this.y = y;
        }
        private String giveCoordinates(){
            return (String.valueOf(x) + "," + String.valueOf(y));
        }
    }
    
    //Constructor for the Graph
    public RouteFinder(){
        size = 100;
        names = new String[size];
        position = new Coordinate[size];
        matrix = new double[size][size];
        numVertices = 0;
    }

    //Method to add a Vertex to the Graph
    public boolean addVertex(String name, int x, int y){
    	//Checking if the Vertex exists already
        int index = indexOf(name);
        if(index!=-1)
        	return false;
        
        //Adding the data into their respective arrays
        names[numVertices] = name;
        position[numVertices] = new Coordinate(x, y);
        numVertices++;
        return true;
    }
    
    //Method to add an edge between 2 vertices in a graph
    public boolean addEdge(String from, String to, double cost){
        int fromIndex = indexOf(from);
        int toIndex = indexOf(to);
        
        //Checking if the vertices exist or not
        if(fromIndex == -1 || toIndex == -1) 
        	return false;
        
        //Checking if an edge exists or not
        if(matrix[fromIndex][toIndex] != 0) 
        	return false;
        
        //Adding the edge cost to the edges matrix
        matrix[fromIndex][toIndex] = cost;
        return true;
    }

    //Method to help print the graph
    public void printGraph(){
        int neighborNum = 0;
        int j = 0;
        for(int i = 0; i < numVertices; i++){
            neighborNum = 0;
            System.out.printf("%d) %s at %s: \n",i+1,names[i],position[i].giveCoordinates());
            for(j = 0; j < numVertices; j++){
                if(matrix[i][j] != 0){
                    neighborNum++;
                    System.out.printf("\t%d) %s at distance: %f\n",neighborNum,names[j],matrix[i][j]);
                } 
            }
            System.out.printf("\n");
        }
    }

    //Method to find the shortest path using A* algorithm
    public void aStarShortestPath(String source, String destination){

    	//Finding the index of the source and destination vertices
        int sourceIndex = indexOf(source);
        int destinationIndex = indexOf(destination);
        if(sourceIndex == -1 || destinationIndex == -1){
            System.out.println("Invalid Inputs!\n");
            return;
        }
            

        //Array to store function values for the search heuristic
        double[] h = new double[numVertices];

        //Array to store the values of the path cost
        double[] g = new double[numVertices];

        int numFinalized = 0;
        
        //Array to represent the finalized vertices (vertex at index 'i' will be finalized if finalized[i] is true)
        boolean[] finalized = new boolean[numVertices];
        
        //Array to store the parent indices of each vertex, this will help us in backtracking
        int[] parents = new int[numVertices];

        //Initialization
        for(int i = 0; i < numVertices; i++){
        	//Cost will be set to infinity
            g[i] = Integer.MAX_VALUE;
            //The parents are unknown
            parents[i] = 0;
            //Calculating the heuristic values for each of the states using a helper method
            h[i] = heurestic(i, destinationIndex);
        }

        /*	Priority Queue to store the Node connections
         *  The NodeConnection class implements the Comparable interface (required for PQ)
         *  The NodeConnection class just holds fields for the index of the node and its current evaluation function value
         */
        PriorityQueue<NodeConnection> pq = new PriorityQueue<NodeConnection>();
        pq.add(new NodeConnection(sourceIndex, 0));
        parents[sourceIndex] = -1;						//Parent of source is set to be -1
        g[sourceIndex] = 0;

        //Looping while the destination index is not finalized
        while(!finalized[destinationIndex] && numFinalized < numVertices){
        	//Polling the node with lowest f-value
            int curr = pq.remove().index;

            if(finalized[curr]){
                continue;
            }

            System.out.printf("Node %s has been finalized\n",names[curr]);

            //Finalizing the node
            finalized[curr] = true;
            numFinalized++;

            //Processing the children of the finalized node
            for(int i = 0; i < numVertices; i++){
            	//If the child has already not been finalized
                if(matrix[curr][i] != 0 && !finalized[i]){
                	//Calculating the new cost
                    double tempCost = g[curr] + matrix[curr][i];
                    //If the new cost is less, then we will update it and add it to the PQ
                    if(tempCost < g[i]){
                        g[i] = tempCost;
                        pq.add(new NodeConnection(i, tempCost + h[i]));
                        parents[i] = curr;
                    }
                }
            }

        }

        //Backtracking to get the path (using stack to reverse it)
        Stack<String> pathList = new Stack<String>();
        int curr = destinationIndex;
        do{
            pathList.push(names[curr]);
            curr = parents[curr];
        } while(curr != -1);
        
        //This part is their to visualize the path
        String str = "";
        while(!pathList.isEmpty())
            str += pathList.pop() + " -> ";
        if(str.length() > 0)
            str = str.substring(0, str.length() - 3);

        System.out.printf("\nThe Shortest path from %s to %s is:\n\t%s\n\tWith a distance of %f\n\n",source,destination,str,g[destinationIndex]);

    }

    //Helper method to find the index of the vertex
    private int indexOf(String name){
        for(int i = 0; i < numVertices; i++){
            if(names[i].equals(name))
                return i;
        }
        return -1;
    }

    //Helper method to calculate the heuristic
    private double heurestic(int curr, int destination){
        double xDiff = position[curr].x - position[destination].x;
        double yDiff = position[curr].y - position[destination].y;
        return Math.sqrt((Math.pow(xDiff, 2))+(Math.pow(yDiff, 2)));
    }

    //Node connection class to store the index and evaluation function value of each node
    private class NodeConnection implements Comparable<NodeConnection>{
        int index;
        double functionValue;
        
        private NodeConnection(int index, double functionValue){
            this.index = index;
            this.functionValue = functionValue;
        }

        public String toString(){
            return "For index " + String.valueOf(index) + " the function vallue is " + String.valueOf(functionValue) + " \n";
        }

        @Override
        public int compareTo(NodeConnection o) {
            return (int)(this.functionValue - o.functionValue);
        }

    }

}