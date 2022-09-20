
public class Main {

	public static void main(String[] args) {
		
		EightPuzzle Demo = new EightPuzzle();
		try {
			Demo.readCommands("/Users/rohansingh/Desktop/P1_1.txt");
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

}
