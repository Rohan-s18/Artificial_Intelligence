import org.junit.*;
import static org.junit.Assert.*;

public class EightPuzzleTester {

	EightPuzzle Dummy;
	
	@Test
	public void setStateTest() {
		//This test will will pass if the program throws an exception when using invalid format/input (eg. Board with 10 states, repeating characters etc.)
		Dummy = new EightPuzzle();
		
		//Trying some invalid states
		try {
			Dummy.setState("b1234567890");
		}
		catch(IllegalArgumentException e) {
			assertTrue(true);
		}
		try {
			Dummy.setState("324");
		}
		catch(IllegalArgumentException e) {
			assertTrue(true);
		}
		try {
			Dummy.setState("b2455555bb");
		}
		catch(IllegalArgumentException e) {
			assertTrue(true);
		}
		try {
			Dummy.setState("asdfgjkl");;
		}
		catch(IllegalArgumentException e) {
			assertTrue(true);
		}
		
		String temp = Dummy.setState("12b654873");
		assertTrue(temp.equals("12b654873"));
		
	}
	
	@Test
	public void maxStates() {
		Dummy = new EightPuzzle();
		//Setting the nodes to be considered as 5
		Dummy.maxNodes(5);
		Dummy.setState("b23751468");
		
		
		//Setting maxNodes to a negative number
		try {
			Dummy.maxNodes(-123);
		}
		catch(IllegalArgumentException e) {
			assertTrue(true);
		}
		
		//Will assert True if we reach the limit 
		try {
			Dummy.solveAStar("h1");
		} catch (Exception e) {
			String temp = e.getMessage();
			assertTrue(temp.equals("Maximum limit has been reached"));
		}
		
	}
	
	@Test
	public void moveTest() {
		Dummy = new EightPuzzle();
		
		//making moves that can't be made due to configuration
		Dummy.move("up");
		Dummy.move("left");
		assertTrue(Dummy.getState().equals("b12345678"));
		
		//Giving valid moves
		Dummy.move("right");
		Dummy.move("down");
		assertTrue(Dummy.getState().equals("1423b5678"));
		
		//Giving inalid moves
		try {
			Dummy.move("Unga Bunga");
		}
		catch(Exception e) {
			assertTrue(true);
		}
		
		
	}
	
}
