package CSDS_391.Search.A_star;

import java.util.*;;

public class A_star {
    private Coordinate[] position;
    private String[] names;
    private double[][] matrix;
    private int size;
    private int numVertices;

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
    
    public A_star(){
        size = 100;
        names = new String[size];
        position = new Coordinate[size];
        matrix = new double[size][size];
        numVertices = 0;
    }

    public boolean addVertex(String name, int x, int y){
        int index = indexOf(name);
        if(index!=-1)return false;
        names[numVertices] = name;
        position[numVertices] = new Coordinate(x, y);
        numVertices++;
        return true;
    }
    
    public boolean addEdge(String from, String to, double cost){
        int fromIndex = indexOf(from);
        int toIndex = indexOf(to);
        if(fromIndex == -1 || toIndex == -1) return false;
        if(matrix[fromIndex][toIndex] != 0) return false;
        matrix[fromIndex][toIndex] = cost;
        return true;
    }

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

    public void aStarShortestPath(String source, String destination){

        int sourceIndex = indexOf(source);
        int destinationIndex = indexOf(destination);
        if(sourceIndex == -1 || destinationIndex == -1){
            System.out.println("Invalid Inputs!\n");
            return;
        }
            

        //Array to store function values for the search heurestic
        double[] h = new double[numVertices];

        //Array to store the values of the path cost
        double[] g = new double[numVertices];

        //Array to store the function values
        //double[] f = new double[numVertices];

        int numFinalized = 0;
        boolean[] finalized = new boolean[numVertices];
        int[] parents = new int[numVertices];

        for(int i = 0; i < numVertices; i++){
            g[i] = Integer.MAX_VALUE;
            //f[i] = Integer.MAX_VALUE;
            parents[i] = 0;
            h[i] = heurestic(i, destinationIndex);
        }

        PriorityQueue<NodeConnection> pq = new PriorityQueue<NodeConnection>();
        pq.add(new NodeConnection(sourceIndex, 0));
        parents[sourceIndex] = -1;
        g[sourceIndex] = 0;
       // f[sourceIndex] = h[sourceIndex];

        while(!finalized[destinationIndex] && numFinalized < numVertices){
            int curr = pq.remove().index;

            if(finalized[curr]){
                continue;
            }

            System.out.printf("Node %s has been finalized\n",names[curr]);

            finalized[curr] = true;
            numFinalized++;

            for(int i = 0; i < numVertices; i++){
                if(matrix[curr][i] != 0 && !finalized[i]){
                    double tempCost = g[curr] + matrix[curr][i];
                    if(tempCost < g[i]){
                        g[i] = tempCost;
                        pq.add(new NodeConnection(i, tempCost + h[i]));
                        parents[i] = curr;
                    }
                }
            }

        }

        Stack<String> pathList = new Stack<String>();
        int curr = destinationIndex;
        do{
            pathList.push(names[curr]);
            curr = parents[curr];
        } while(curr != -1);
        
        String str = "";
        while(!pathList.isEmpty())
            str += pathList.pop() + " -> ";
        if(str.length() > 0)
            str = str.substring(0, str.length() - 3);

        System.out.printf("\nThe Shortest path from %s to %s is:\n\t%s\n\tWith a distance of %f\n\n",source,destination,str,g[destinationIndex]);

    }


    private int indexOf(String name){
        for(int i = 0; i < numVertices; i++){
            if(names[i].equals(name))
                return i;
        }
        return -1;
    }

    private double heurestic(int curr, int destination){
        double xDiff = position[curr].x - position[destination].x;
        double yDiff = position[curr].y - position[destination].y;
        return Math.sqrt((Math.pow(xDiff, 2))+(Math.pow(yDiff, 2)));
    }

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
