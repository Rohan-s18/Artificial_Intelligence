public class HC extends Graph{
    
    public void hillClimb(String source){
        int sourceIndex = indexOf(source);
        if(sourceIndex == -1){
            System.out.println("Invalid source");
            return;
        }

        int currMax = values[sourceIndex];
        int currMaxIndex = sourceIndex;

        boolean foundMax = false;

        int temp = 0;

        while(!foundMax){
            temp = getGreatestNeighbor(currMaxIndex);
            if(temp == -1)
                break;
            if(values[temp] >= currMax){
                currMax = values[temp];
                currMaxIndex = temp;
            }
        }

        System.out.printf("\nThe local maxima is node: %s (%d)\n\n",names[currMaxIndex],currMax);


    }

    private int getGreatestNeighbor(int index){
        int max = values[index];
        int maxIndex = index;
        for(int i = 0; i < numVertices; i++){
            if(matrix[index][i] != 0 && values[i] > max){
                max = values[i];
                maxIndex = i;
            }
        }
        if(maxIndex == index)
            return -1;
        return maxIndex;
    }

}
