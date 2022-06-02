package CSDS_391.Search.Optimization;

public class Demo {
    
    public static void main(String[] args) {
        simulatedAnnealingDemo();
    }

    public static void hillClimbDemo(){

        HC demoGraph = new HC();

        demoGraph.addVertex("A", 10);
        demoGraph.addVertex("B", 50);
        demoGraph.addVertex("C", 13);
        demoGraph.addVertex("D", 17);
        demoGraph.addVertex("E", 10);
        demoGraph.addVertex("F", 10);
        demoGraph.addVertex("G", 10);
        demoGraph.addVertex("H", 10);
        demoGraph.addVertex("I", 10);
        demoGraph.addVertex("J", 10);
        demoGraph.addVertex("K", 10);
        demoGraph.addVertex("L", 10);
        demoGraph.addVertex("M", 10);
        demoGraph.addVertex("N", 10);
        demoGraph.addVertex("O", 10);
        demoGraph.addVertex("P", 10);
        demoGraph.addVertex("Q", 10);
        demoGraph.addVertex("R", 10);
        demoGraph.addVertex("S", 10);
        demoGraph.addVertex("T", 10);

        demoGraph.addEdge("A", "C", 1);
        demoGraph.addEdge("A", "D", 1);

        demoGraph.addEdge("B", "D", 1);

        demoGraph.addEdge("C", "A", 1);
        demoGraph.addEdge("C", "D", 1);

        demoGraph.addEdge("D", "A", 1);
        demoGraph.addEdge("D", "B", 1);
        demoGraph.addEdge("D", "C", 1);
        demoGraph.addEdge("D", "E", 1);
        demoGraph.addEdge("D", "I", 1);

        demoGraph.addEdge("E", "D", 1);
        demoGraph.addEdge("E", "J", 1);

        demoGraph.addEdge("F", "G", 1);
        demoGraph.addEdge("F", "H", 1);

        demoGraph.addEdge("G", "F", 1);
        demoGraph.addEdge("G", "I", 1);

        demoGraph.addEdge("H", "F", 1);
        demoGraph.addEdge("H", "I", 1);

        demoGraph.addEdge("I", "D", 1);
        demoGraph.addEdge("I", "G", 1);
        demoGraph.addEdge("I", "H", 1);
        demoGraph.addEdge("I", "J", 1);

        demoGraph.addEdge("J", "E", 1);
        demoGraph.addEdge("J", "I", 1);
        demoGraph.addEdge("J", "M", 1);

        demoGraph.addEdge("K", "L", 1);

        demoGraph.addEdge("L", "K", 1);
        demoGraph.addEdge("L", "M", 1);
        demoGraph.addEdge("L", "Q", 1);

        demoGraph.addEdge("M", "J", 1);
        demoGraph.addEdge("M", "L", 1);
        demoGraph.addEdge("M", "R", 1);

        demoGraph.addEdge("N", "H", 1);
        demoGraph.addEdge("N", "O", 1);
        demoGraph.addEdge("N", "S", 1);

        demoGraph.addEdge("O", "N", 1);
        demoGraph.addEdge("O", "P", 1);
        demoGraph.addEdge("O", "T", 1);

        demoGraph.addEdge("P", "O", 1);
        demoGraph.addEdge("P", "Q", 1);

        demoGraph.addEdge("Q", "L", 1);
        demoGraph.addEdge("Q", "P", 1);

        demoGraph.addEdge("R", "M", 1);

        demoGraph.addEdge("S", "N", 1);
        demoGraph.addEdge("S", "T", 1);

        demoGraph.addEdge("T", "S", 1);
        demoGraph.addEdge("T", "O", 1);

        demoGraph.hillClimb("A");

    }


    public static void simulatedAnnealingDemo(){
        SimulatedAnnealing demoGraph = new SimulatedAnnealing();

        demoGraph.addVertex("A", 14);
        demoGraph.addVertex("B", 15);
        demoGraph.addVertex("C", 13);
        demoGraph.addVertex("D", 11);
        demoGraph.addVertex("E", 10);
        demoGraph.addVertex("F", 12);
        demoGraph.addVertex("G", 17);
        demoGraph.addVertex("H", 16);
        demoGraph.addVertex("I", 17);
        demoGraph.addVertex("J", 18);
        demoGraph.addVertex("K", 13);
        demoGraph.addVertex("L", 19);
        demoGraph.addVertex("M", 11);
        demoGraph.addVertex("N", 10);
        demoGraph.addVertex("O", 10);
        demoGraph.addVertex("P", 19);
        demoGraph.addVertex("Q", 7);
        demoGraph.addVertex("R", 8);
        demoGraph.addVertex("S", 15);
        demoGraph.addVertex("T", 18);

        demoGraph.addEdge("A", "C", 1);
        demoGraph.addEdge("A", "D", 1);

        demoGraph.addEdge("B", "D", 1);

        demoGraph.addEdge("C", "A", 1);
        demoGraph.addEdge("C", "D", 1);
        demoGraph.addEdge("C", "G", 1);

        demoGraph.addEdge("D", "A", 1);
        demoGraph.addEdge("D", "B", 1);
        demoGraph.addEdge("D", "C", 1);
        demoGraph.addEdge("D", "E", 1);
        demoGraph.addEdge("D", "I", 1);

        demoGraph.addEdge("E", "D", 1);
        demoGraph.addEdge("E", "J", 1);

        demoGraph.addEdge("F", "G", 1);
        demoGraph.addEdge("F", "H", 1);

        demoGraph.addEdge("G", "C", 1);
        demoGraph.addEdge("G", "F", 1);
        demoGraph.addEdge("G", "I", 1);

        demoGraph.addEdge("H", "F", 1);
        demoGraph.addEdge("H", "I", 1);

        demoGraph.addEdge("I", "D", 1);
        demoGraph.addEdge("I", "G", 1);
        demoGraph.addEdge("I", "H", 1);
        demoGraph.addEdge("I", "J", 1);

        demoGraph.addEdge("J", "E", 1);
        demoGraph.addEdge("J", "I", 1);
        demoGraph.addEdge("J", "M", 1);

        demoGraph.addEdge("K", "L", 1);

        demoGraph.addEdge("L", "K", 1);
        demoGraph.addEdge("L", "M", 1);
        demoGraph.addEdge("L", "Q", 1);

        demoGraph.addEdge("M", "J", 1);
        demoGraph.addEdge("M", "L", 1);
        demoGraph.addEdge("M", "R", 1);

        demoGraph.addEdge("N", "H", 1);
        demoGraph.addEdge("N", "O", 1);
        demoGraph.addEdge("N", "S", 1);

        demoGraph.addEdge("O", "N", 1);
        demoGraph.addEdge("O", "P", 1);
        demoGraph.addEdge("O", "T", 1);

        demoGraph.addEdge("P", "O", 1);
        demoGraph.addEdge("P", "Q", 1);

        demoGraph.addEdge("Q", "L", 1);
        demoGraph.addEdge("Q", "P", 1);

        demoGraph.addEdge("R", "M", 1);

        demoGraph.addEdge("S", "N", 1);
        demoGraph.addEdge("S", "T", 1);

        demoGraph.addEdge("T", "S", 1);
        demoGraph.addEdge("T", "O", 1);

        //demoGraph.findStochasticMax("A");

        int temp = 0, ct = 0;

        while(temp != 19 && ct < 1000){
            temp = demoGraph.findStochasticMax("A");
            ct++;
        }
        if(temp == 19)
            System.out.printf("\nFound the global maximum at the %dth iteration\n\n",ct);
        else
            System.out.printf("\nCouldn't find the global maximum\n\n");
        

    }


}
