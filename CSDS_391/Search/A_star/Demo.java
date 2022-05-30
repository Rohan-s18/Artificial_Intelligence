package CSDS_391.Search.A_star;

public class Demo {
    public static void main(String[] args) {

        


    }

    public void demoGraph(){
        //System.out.println("\n\nHello World\n\n");
        
        Graph demoGraph = new Graph();
        
        //Adding Vertices
        demoGraph.addVertex("A");
        demoGraph.addVertex("B");
        demoGraph.addVertex("C");
        demoGraph.addVertex("D");
        demoGraph.addVertex("L");
        demoGraph.addVertex("E");
        demoGraph.addVertex("F");
        demoGraph.addVertex("G");
        demoGraph.addVertex("H");
        demoGraph.addVertex("I");

        //Adding Edges
        demoGraph.addEdge("A", "B", 1);
        demoGraph.addEdge("A", "C", 10);

        demoGraph.addEdge("B", "C", 5);

        demoGraph.addEdge("A", "L", 1);
        demoGraph.addEdge("L", "C", 23);
        demoGraph.addEdge("L", "E", 12);

        demoGraph.addEdge("C", "D", 3);
        demoGraph.addEdge("C", "F", 3);

        demoGraph.addEdge("D", "F", 1);
        demoGraph.addEdge("D", "G", 4);
        demoGraph.addEdge("D", "I", 15);

        demoGraph.addEdge("F", "E", 7);
        demoGraph.addEdge("F", "H", 20);

        demoGraph.addEdge("G", "D", 4);
        demoGraph.addEdge("G", "H", 2);

        demoGraph.addEdge("H", "F", 2);
        demoGraph.addEdge("H", "G", 20);

        demoGraph.removeVertex("L");

        //Printing the graph
        demoGraph.printGraph();
    }

    public void demoAStarGraph(){
        //System.out.println("\n\nHello World\n\n");
        
        A_star demoGraph = new A_star();
        
        //Adding Vertices
        demoGraph.addVertex("A",0,0);
        demoGraph.addVertex("B",-1,0);
        demoGraph.addVertex("C",-1,5);
        demoGraph.addVertex("D",-1,7);
        demoGraph.addVertex("E",-7,5);
        demoGraph.addVertex("F",-2,-5);
        demoGraph.addVertex("G",-1,10);
        demoGraph.addVertex("H",-1,8);
        demoGraph.addVertex("I",9,7);

        //Adding Edges
        demoGraph.addEdge("A", "B", 1);
        demoGraph.addEdge("A", "C", 10);

        demoGraph.addEdge("B", "C", 5);

        demoGraph.addEdge("C", "D", 3);
        demoGraph.addEdge("C", "F", 3);

        demoGraph.addEdge("D", "F", 1);
        demoGraph.addEdge("D", "G", 4);
        demoGraph.addEdge("D", "I", 15);

        demoGraph.addEdge("F", "E", 7);
        demoGraph.addEdge("F", "H", 20);

        demoGraph.addEdge("G", "D", 4);
        demoGraph.addEdge("G", "H", 2);

        demoGraph.addEdge("H", "F", 2);
        demoGraph.addEdge("H", "G", 20);

        //Printing the graph
        demoGraph.printGraph();
    }
}
