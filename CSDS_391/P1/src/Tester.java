
public class Tester {

	public static void main(String[] args) {
		
		EightPuzzle foo = new EightPuzzle();
		//foo.printBoard();
		
		/*foo.move("right");
		foo.move("right");
		foo.move("down");*/
		
		foo.randomizeState(100);
		
		foo.printBoard();
		
		foo.maxNodes(15);
		
		
		try {
			foo.solveAStar("h2");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
