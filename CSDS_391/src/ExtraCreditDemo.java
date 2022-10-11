
public class ExtraCreditDemo {

	public static void main(String[] args) {
		RouteFinder demoGraph = new RouteFinder();
        
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
        //demoGraph.printGraph();

        demoGraph.aStarShortestPath("A", "I");

	}

}
