
public class Experiment {

	public static void main(String[] args) {
		
		EightPuzzle foo = new EightPuzzle();
		
		//foo.randomizeState(10000);
		foo.maxNodes(1000);
		foo.setState("b34572168");
		
		try {
			//long t1 = System.nanoTime();
			//foo.solveAStar("h1");
			foo.solveBeam(5);
			//long t2 = System.nanoTime();
			//System.out.printf("The runtime was %d milliseconds\n",(t2-t1)/1000000);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//foo.printBoard();
		
		/*foo.move("right");
		foo.move("right");
		foo.move("down");
		
		//foo.randomizeState(1000);
		
		foo.setState("b23751468");
		
		foo.printBoard();
		
		
		try {
			//foo.solveAStar("h2");
			//foo.solveAStar("h1");
			foo.solveBeam(3);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		System.out.println("\n\n\n");
		
		*/
		
		/*
		foo.randomizeState(100);
		
		foo.printBoard();
		
		
		foo.solveBeam(15);
		*/
		
		//State: b23751468
		//h1: We solved the puzzle in 19 moves and considered 1031 nodes
		//h2: We solved the puzzle in 23 moves and considered 315 nodes
		//Local Beam: k = 1, We couldn't solve the puzzle and we used 0 moves and considered 135 nodes
		//Local Beam, k = 2, We couldn't solve the puzzle and we used 0 moves and considered 53 nodes
		//Local Beam, k = 3, We solved the puzzle in 95 moves and considered 94 nodes
		//Local Beam, k = 4, We solved the puzzle in 59 moves and considered 58 nodes
		//Local Beam, k = 5, We solved the puzzle in 41 moves and considered 40 nodes
		//Local Beam, k = 10, We solved the puzzle in 23 moves and considered 22 nodes
		//Local Beam, k = 25, We solved the puzzle and we used 21 moves and considered 20 nodes
		//Local Beam, k = 100, We did not solve the puzzle, we used 19 moves and considered 18 nodes
		//Local Beam, k = infinity, We solved the puzzle in 19 moves and considered 18 nodes
		
		
		
		
		
		
	}

}
