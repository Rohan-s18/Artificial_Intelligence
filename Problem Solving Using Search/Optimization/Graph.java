public class Graph{
    int[][] matrix; 
    String[] names;
    int numVertices;
    int size;

    public Graph(){
        matrix = new int[100][100];
        names = new String[100];
        size = 100;
        numVertices = 0;
    }

    public boolean addVertex(String name){
        if(indexOf(name)!=-1)
            return false;
        if(numVertices>=size)
            resize();
        names[numVertices] = name;
        numVertices++;
        return true;
    }

    public boolean addEdge(String from, String to, int cost){
        int fromIndex = indexOf(from);
        int toIndex = indexOf(to);
        if(toIndex == -1 || fromIndex == -1)
            return false;
        matrix[fromIndex][toIndex] = cost;
        return true;
    }

    public boolean removeEdge(String from, String to){
        int fromIndex = indexOf(from);
        int toIndex = indexOf(to);
        if(fromIndex == -1 || toIndex == -1)
            return false;
        if(matrix[fromIndex][toIndex]==0)
            return false;
        matrix[fromIndex][toIndex] = 0;
        return true;
    }

    public boolean removeVertex(String name){
        int index = indexOf(name);
        if(index == -1)
            return false;
        for(int i = index; i < numVertices - 1; i++)
            names[i] = names[i+1];
        for(int j = 0; j < numVertices; j++){
            for(int k = index; k < numVertices - 1; k++)
                matrix[j][k] = matrix[j][k+1];
        }
        for(int l = index; l < numVertices - 1; l++)
            matrix[l] = matrix[l+1];
        numVertices--;
        return true;
    }

    public void printGraph(){
        int neighborNum = 0;
        int j = 0;
        for(int i = 0; i < numVertices; i++){
            neighborNum = 0;
            System.out.printf("%d) %s: \n",i+1,names[i]);
            for(j = 0; j < numVertices; j++){
                if(matrix[i][j] != 0){
                    neighborNum++;
                    System.out.printf("\t%d) %s at distance: %d\n",neighborNum,names[j],matrix[i][j]);
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

    private void resize(){
        int[][] oldMatrix = matrix;
        String[] oldNames = names;
        int oldSize = size;
        size *= 2;
        matrix = new int[size][size];
        names = new String[size];
        for(int i = 0; i < oldSize; i++){
            names[i] = oldNames[i];
            for(int j = 0; j < oldSize; j++)
                matrix[i][j] = oldMatrix[i][j];
        }
    }

}