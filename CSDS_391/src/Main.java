
public class Main {

	public static void main(String[] args) {
		
		EightPuzzle Demo = new EightPuzzle();
		
		String filepath_1 = "/Users/rohansingh/Desktop/P1.txt";
		String filepath_2 = "/Users/rohansingh/Desktop/P1_1.txt";
		String filepath_3 = "/Users/rohansingh/Desktop/P1_2.txt";
		
		
		try {
			Demo.readCommands(filepath_3);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

	// b65 793 428
	// b65793428
	
}
