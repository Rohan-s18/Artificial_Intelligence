import java.util.*;
import java.io.*;

public class EightPuzzle {
	
	//Private BoardState inner class at the bottom of the code
	private BoardState state;			//Instance variable that refers to the current board state
	private int maxStates;				//This holds the maximum number of nodes that can be expanded
	
	//Constructor
	public EightPuzzle() {
		//Cerating the initial state
		state = new BoardState("b12345678", 0, null);
		maxStates = Integer.MAX_VALUE;		//If not specified, setting the maximum number of expandable nodes to the max Integer 
	}
	
	//Method to set the state of the board to the String state that is provided in the text.
	public boolean setState(String state) {
		
		//Checking if the provided String is valid or not using an isValid() helper method
		if(!isValidState(state))
			return false;
		
		//Setting the String state 
		this.state.myState = state;
		
		//Setting the new Blank Tile position
		this.state.myBlankTile = getBlankTile();
		return true;
	}
	
	//Printing the state of the board
	public void printState() {
		System.out.println(state.myState);
	}
	
	
	//Setting the maximum number of expandable nodes
	public void maxNodes(int n) {
		maxStates = n;
	}
	
	//Method to make the move
	public boolean move(String move) {
		
		//The case that the command is given as "up"
		if(move.equals("up")) {
			//Calling the moveUp helper method
			this.state = moveUp(state);
			return true;
		}
		
		//The case that the command is given as "down"
		if(move.equals("down")) {
			//Calling the moveDown helper method
			this.state = moveDown(state);
			return true;
		}
		
		//The case that the command is given as "left"
		if(move.equals("left")) {
			//Calling the moveLeft helper method
			this.state = moveLeft(state);
			return true;
		}
		
		//The case that the command is given as "right"
		if(move.equals("right")) {
			//Calling the moveRight helper method
			this.state = moveRight(state);
			return true;
		}
		
		//If a different String argument was given
		return false;
	}
	
	//Method to randomize the state of the 8-puzzle
	public void randomizeState(int n) {
		//Resetting to the goal position
		state.myState = "b12345678";
		state.myBlankTile = 0;
		
		//Array of the legal moves that we can make
		String[] legalMoves = {"up","down","left","right"};
		String tempMove;
		for(int i = 0; i < n; i++) {
			//Getting a random move
			tempMove = legalMoves[(int)(4*Math.random())];
			move(tempMove);
		}
	}
	
	//Method to solve the 8-puzzle using the A* algorithm
	public void solveAStar(String heuristic) throws Exception {
		//Going to the solveH1() helper method if the heuristic provided is h1
		if(heuristic.equals("h1"))
			solveH1();
		//Going to the solveH1() helper method if the heuristic provided is h2
		else if(heuristic.equals("h2"))
			solveH2();
		else
			throw new Exception("Invalid heurestic provided");
	}
	
	
	//Private helper methods
	
	private void solveH1() {
		//Keeping the cost under f
		//int d = 1;
		//int f = 0; 
		
		PriorityQueue<BoardState> pq = new PriorityQueue<BoardState>();
		HashSet<String> finalized = new HashSet<String>();
		List<BoardState> myBoardList = new ArrayList<BoardState>();
		
		pq.add(new BoardState(state.myState,0,null));
		
		int ct = 0;
		while(!finalized.contains("b12345678") && !pq.isEmpty() && ct < maxStates) {
			BoardState curr = pq.remove();
			
			if(finalized.contains(curr.myState))
				continue;
			
			System.out.printf("The state: %s has been finalized!\n",curr.myState);
			finalized.add(curr.myState);
			myBoardList.add(curr);
			
			List<BoardState> tempList = curr.generateChildrenH1();
			for(int i = 0; i < tempList.size(); i++) {
				BoardState temp = tempList.get(i);
				if(!finalized.contains(temp.myState)) {
					pq.add(temp);
				}
			}
			
			ct++;
		}
		
		if(ct == maxStates)
			System.out.println("Maximum limit has been reached");
		
		BoardState destination = null;
		if(finalized.contains("b12345678")) {
			System.out.println("We solved the puzzle yay!");
			for(int i = 0; i < myBoardList.size(); i++) {
				if(myBoardList.get(i).myState.equals("b12345678"))
					destination = myBoardList.get(i);
			}
		}
		
		Stack<BoardState> path = new Stack<BoardState>();
		while(destination != null) {
			path.push(destination);
			destination = destination.parentState;
		}
		
		while(!path.isEmpty()) {
			printBoardState(path.pop().myState);
		}
		
	}
	
	private void solveH2() {
		PriorityQueue<BoardState> pq = new PriorityQueue<BoardState>();
		HashSet<String> finalized = new HashSet<String>();
		List<BoardState> myBoardList = new ArrayList<BoardState>();
		
		pq.add(new BoardState(state.myState,0,null));
		
		int ct = 0;
		while(!finalized.contains("b12345678") && !pq.isEmpty() && ct < maxStates) {
			BoardState curr = pq.remove();
			
			if(finalized.contains(curr.myState))
				continue;
			
			//System.out.printf("The state: %s has been finalized!\n",curr.myState);
			finalized.add(curr.myState);
			myBoardList.add(curr);
			
			List<BoardState> tempList = curr.generateChildrenH2();
			for(int i = 0; i < tempList.size(); i++) {
				BoardState temp = tempList.get(i);
				if(!finalized.contains(temp.myState)) {
					pq.add(temp);
				}
			}
			
			ct++;
		}
		
		if(ct == maxStates)
			System.out.println("Maximum limit has been reached");
		
		BoardState destination = null;
		if(finalized.contains("b12345678")) {
			System.out.println("We solved the puzzle yay!\n");
			for(int i = 0; i < myBoardList.size(); i++) {
				if(myBoardList.get(i).myState.equals("b12345678"))
					destination = myBoardList.get(i);
			}
		}
		
		Stack<BoardState> path = new Stack<BoardState>();
		while(destination != null) {
			path.push(destination);
			destination = destination.parentState;
		}
		
		while(!path.isEmpty()) {
			printBoardState(path.pop().myState);
		}
	}
	
	private int h1(String myState) {
		int misplaced = 0;
		char[] tiles = {'b','1','2','3','4','5','6','7','8'};
		for(int i = 0; i < 9; i++) {
			if(myState.charAt(i) != tiles[i])
				misplaced++;
		}
		return misplaced;
	}
	
	private int h2(String tempState) {
		int distanceSum = 0;
		char temp;
		for(int i = 0; i < 9; i++) {
			temp = tempState.charAt(i);
			if(temp == 'b') 
				distanceSum += (i-0);
			else
				distanceSum += Math.abs(i-(temp-48));
		}
		return distanceSum;
	}
	
	private BoardState moveUp(BoardState myState) {
		if(myState.myBlankTile < 3) {
			return myState;
		}
		myState = swap(myState.myBlankTile, myState.myBlankTile - 3, myState);
		myState.myBlankTile -= 3;
		return myState;
	}
	
	private BoardState moveDown(BoardState myState) {
		if(myState.myBlankTile >= 6) {
			return myState;
		}
		myState = swap(myState.myBlankTile, myState.myBlankTile + 3, myState);
		myState.myBlankTile += 3;
		return myState;
	}

	private BoardState moveLeft(BoardState myState) {
		if(myState.myBlankTile%3 == 0) {
			return myState;
		}
		myState = swap(myState.myBlankTile, myState.myBlankTile - 1,myState);
		myState.myBlankTile--;
		return myState;
	}

	private BoardState moveRight(BoardState myState) {
		if((myState.myBlankTile+1)%3 == 0) {
			return myState;
		}
		myState = swap(myState.myBlankTile, myState.myBlankTile + 1,myState);
		myState.myBlankTile++;
		return myState;
	}
	
	private BoardState swap(int i, int j, BoardState myState) {
		String tempState = myState.myState;
		char c1 = tempState.charAt(i);
		char c2 = tempState.charAt(j);
		StringBuilder temp = new StringBuilder();
		for(int index = 0; index < 9; index++) {
			if(index == i) {
				temp.append(c2);
			} 
			else if(index == j) {
				temp.append(c1);
			}
			else {
				temp.append(tempState.charAt(index));
			}	
		}
		tempState = temp.toString();
		myState.myState = tempState;
		return myState;
	}
	
	private int getBlankTile() {
		for(int i = 0; i < 9; i++) {
			if(state.myState.charAt(i) == 'b')
				return i;
		}
		return 0;
	}
	
	
	public void readCommands(String filepath) throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(new File(filepath)));
		
		String line;
		
		while((line = br.readLine()) != null) {
			if(line.substring(0, 8).equals("setState")) {
				setState(line.substring(9));
			}
			else if(line.equals("printState")) {
				printBoardState(state.myState);
			}
			else if(line.substring(0, 4).equals("move")) {
				move(line.substring(5));
			}
			else if(line.substring(0,14).equals("randomizeState")) {
				randomizeState(Integer.parseInt(line.substring(15)));
			}
			else if(line.substring(0, 12).equals("solve A-star")) {
				solveAStar(line.substring(13));
			}
			else if(line.substring(0, 10).equals("solve beam")) {
				solveBeam(Integer.parseInt(line.substring(11)));
			}
			else if(line.substring(0, 8).equals("maxNodes")) {
				maxNodes(Integer.parseInt(line.substring(9)));
			}
		}
		
	}
	
	private boolean isValidState(String state) {
		if(state.length() != 9)
			return false;
		
		List<Character> tileList = new ArrayList<Character>();
		tileList.add('b');
		tileList.add('1');
		tileList.add('2');
		tileList.add('3');
		tileList.add('4');
		tileList.add('5');
		tileList.add('6');
		tileList.add('7');
		tileList.add('8');
		
		try {
			char temp;
			for(int i = 0; i < 9; i++) {
				temp = state.charAt(i);
				if(!tileList.contains(temp))
					return false;
				int index = tileList.indexOf(temp);
				tileList.remove(index);
			}
		}
		catch(Exception e) {
			return false;
		}
		
		
		return true;
	}
	
	public void printBoard() {
		System.out.printf("%s\n", state.myState.substring(0, 3));
		System.out.printf("%s\n", state.myState.substring(3, 6));
		System.out.printf("%s\n", state.myState.substring(6, 9));
		System.out.println();
	}
	
	public void solveBeam(int k) {
		
	}
	
	private class BoardState implements Comparable<BoardState>{
		int functionValue;
		String myState;
		BoardState parentState;
		int myBlankTile;
		
		private BoardState(String myState, int functionValue, BoardState parentState) {
			this.functionValue = functionValue;
			this.myState = myState;
			this.parentState = parentState;
			for(int i = 0; i < 9; i++) {
				if(myState.charAt(i) == 'b')
					myBlankTile = i;
			}
		}
		
		public int compareTo(BoardState o) {
			return this.functionValue - o.functionValue;
		}
		
		public List<BoardState> generateChildrenH1(){
			List<BoardState> childrenList = new ArrayList<BoardState>();
			BoardState temp = new BoardState(this.myState,this.functionValue,this.parentState);
			
			//Up Child
			temp = moveUp(temp);
			if(!temp.myState.equals(this.myState)) {
				int depth = functionValue - h1(myState) + 1;
				childrenList.add(new BoardState(temp.myState,depth + h1(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			
			//Down Child
			temp = moveDown(temp);
			if(!temp.myState.equals(this.myState)) {
				int depth = functionValue - h1(myState) + 1;
				childrenList.add(new BoardState(temp.myState,depth + h1(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			//Left Child
			temp = moveLeft(temp);
			if(!temp.myState.equals(this.myState)) {
				int depth = functionValue - h1(myState) + 1;
				childrenList.add(new BoardState(temp.myState,depth + h1(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			//Right Child
			temp = moveRight(temp);
			if(!temp.myState.equals(this.myState)) {
				int depth = functionValue - h1(myState) + 1;
				childrenList.add(new BoardState(temp.myState,depth + h1(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			return childrenList;
		}
		
		public List<BoardState> generateChildrenH2(){
			List<BoardState> childrenList = new ArrayList<BoardState>();
			BoardState temp = new BoardState(this.myState,this.functionValue,this.parentState);
			
			//Up Child
			temp = moveUp(temp);
			if(!temp.myState.equals(this.myState)) {
				int depth = functionValue - h2(myState) + 1;
				childrenList.add(new BoardState(temp.myState,depth + h2(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			
			//Down Child
			temp = moveDown(temp);
			if(!temp.myState.equals(this.myState)) {
				int depth = functionValue - h2(myState) + 1;
				childrenList.add(new BoardState(temp.myState,depth + h2(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			//Left Child
			temp = moveLeft(temp);
			if(!temp.myState.equals(this.myState)) {
				int depth = functionValue - h2(myState) + 1;
				childrenList.add(new BoardState(temp.myState,depth + h2(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			//Right Child
			temp = moveRight(temp);
			if(!temp.myState.equals(this.myState)) {
				int depth = functionValue - h2(myState) + 1;
				childrenList.add(new BoardState(temp.myState,depth + h2(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			return childrenList;
		}
		
	}
	
	
	
	
	private void printBoardState(String tempState) {
		System.out.printf("%s\n", tempState.substring(0, 3));
		System.out.printf("%s\n", tempState.substring(3, 6));
		System.out.printf("%s\n", tempState.substring(6, 9));
		System.out.println();
	}

}
