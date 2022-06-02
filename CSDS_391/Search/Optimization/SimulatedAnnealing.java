package CSDS_391.Search.Optimization;

import java.util.*;

public class SimulatedAnnealing extends Graph {
   
    public int findStochasticMax(String source){
        int sourceIndex = indexOf(source);
        if(sourceIndex == -1){
            System.out.println("Invalid source");
            return -1;
        }

        int currMax = values[sourceIndex];
        int currMaxIndex = sourceIndex;

        int tempIndex = 0;
        double temperature = 10;

        while(true){
            tempIndex = chooseNeighbor(currMaxIndex, temperature);
            if(tempIndex == -1)
                break;
            currMax = values[tempIndex];
            currMaxIndex = tempIndex;
            if(temperature != 1)
                temperature -= 1;
        }

        //System.out.printf("\nThe local maxima is node: %s (%d)\n\n",names[currMaxIndex],currMax);

        return currMax;


    }
    
    private int chooseNeighbor(int index, double temperature){
        int currMax = values[index];
        int currMaxIndex = index;
        List<Integer> myList = new ArrayList<Integer>();
        
        for(int i = 0; i < numVertices; i++){
            if(matrix[index][i] != 0){
                myList.add(i);
                if(values[i] >= currMax){
                    currMax = values[i];
                    currMaxIndex = i;
                }
            }
        }

        if(upOrDown()){
            for(int i = 0; i < myList.size(); i++){
                if(chooseThis(values[myList.get(i)],currMax,temperature))
                    return myList.get(i);
            }
        }
        else{
            for(int i = myList.size() - 1; i >= 0; i--){
                if(chooseThis(values[myList.get(i)],currMax,temperature))
                    return myList.get(i);
            }
        }
            
        if(currMaxIndex != index)
            return currMaxIndex;
        return -1;
    }

    private boolean chooseThis(int currValue, int maxValue, double temperature){
        int diff = currValue - maxValue;
        double pow = ((double)diff)/temperature;
        double probabilty = Math.pow(Math.E, pow);
        if(Math.random() < probabilty)
            return true;
        return false;
    }

    private boolean upOrDown(){
        return Math.random() >= 0.5;
    }

}
