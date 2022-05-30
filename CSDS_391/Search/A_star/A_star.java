package CSDS_391.Search.A_star;

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

}
