import java.util.*;
import java.io.*;

//Class that represents an eight puzzle
public class EightPuzzle {
	
	//Private BoardState inner class at the bottom of the code
	private BoardState state;			//Instance variable that refers to the current board state
	private int maxStates;				//This holds the maximum number of nodes that can be expanded
	
	//Constructor
	public EightPuzzle() {
		//Creating the initial state
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
	
	//Method to solve the puzzle using Local Beam search using k states
	public void solveBeam(int k) {
		//Initializing the variables
		List<BoardState> myList = new ArrayList<BoardState>();			//List to keep track of the states
		PriorityQueue<BoardState> bestChildren = new PriorityQueue<BoardState>();		//Priority queue to find the best states
		HashSet<String> finalized = new HashSet<String>();				//Keeping a HashSet to keep track of the finalized states
		BoardState goalState = null;
		
		//Adding the current state to the list
		myList.add(this.state);
		
		int ct = 0;														//Counting the number of nodes that are being ocnsidered
		boolean found = false;											//Boolean flag to see if we have reached the goal state or not
		while(!myList.isEmpty() && !found && ct < maxStates) {
			//Reinitializing the priority queue
			bestChildren = new PriorityQueue<BoardState>();
			
			//iterating through the list of 'k' states
			for(int i = 0; i < myList.size(); i++) {
				//Getting the state, finalizing it, and then generating the list of its children
				BoardState curr = myList.get(i);
				finalized.add(curr.myState);
				List<BoardState> tempList = curr.generateChildrenLocalBeam();
				
				//Iterating through its children and adding them to the Priority Queue
				for(int j = 0; j < tempList.size(); j++) {
					BoardState temp = tempList.get(j);
					//If the child has been finalized, then we will be moving on
					if(finalized.contains(temp.myState))
						continue;
					bestChildren.add(temp);
				}
				
			}
			
			//Re-initializing the beam of states
			myList = new ArrayList<BoardState>();
			
			//adding the 'k' most optimum states to the list
			for(int i = 0; i < k && !bestChildren.isEmpty(); i++) {
				myList.add(bestChildren.remove());
				//If we find the goalState, then we change the flag to found and set the goalState equal to it 
				if(myList.get(i).myState.equals("b12345678")) {
					found = true;
					goalState = myList.get(i);
				}
			}
			
			ct++;
		}
		
		//If we exited because of the Maximum limit being reached
		if(ct == maxStates)
			System.out.println("Maximum limit has been reached");
			
		
		if(goalState != null)
			System.out.println("We solved the puzzle!");
		
		
		//Getting the path by backtracking
		Stack<BoardState> path = new Stack<BoardState>();
		while(goalState != null) {
			path.push(goalState);
			goalState = goalState.parentState;
		}
		
		//Printing the boards that we get in the path
		while(!path.isEmpty()) {
			printBoardState(path.pop().myState);
		}
		
	}
	
	//Private helper methods
	private void solveH1() {
		//Initializing the variables
		PriorityQueue<BoardState> pq = new PriorityQueue<BoardState>();		//Using the priority queue to find the lowest cost node
		HashSet<String> finalized = new HashSet<String>();					//Keeping the set of finalized states
		List<BoardState> myBoardList = new ArrayList<BoardState>();			
		
		//Adding the source board state to the priority queue
		pq.add(new BoardState(state.myState,0,null));
		
		int ct = 0;
		//While the finalized set doesn't contain the goal state, the priority queue is not empty and the number of expanded nodes is less than the maxStates
		while(!finalized.contains("b12345678") && !pq.isEmpty() && ct < maxStates) {
			//Getting the lowest cost State
			BoardState curr = pq.remove();
			
			if(finalized.contains(curr.myState))
				continue;
			
			//System.out.printf("The state: %s has been finalized!\n",curr.myState);
			//Adding the state to the set of finalized states
			finalized.add(curr.myState);
			myBoardList.add(curr);
			
			//Generating the children of the current BoardState using the generateChildren method of the BoardState class
			List<BoardState> tempList = curr.generateChildrenH1();
			for(int i = 0; i < tempList.size(); i++) {
				BoardState temp = tempList.get(i);
				
				//Adding the child to the priority queue if it hasn't been finalized yet
				if(!finalized.contains(temp.myState)) {
					pq.add(temp);
				}
			}
			
			//Increasing the count
			ct++;
		}
		
		//If we exited because of the Maximum limit being reached
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
		
		//Getting the path by backtracking
		Stack<BoardState> path = new Stack<BoardState>();
		while(destination != null) {
			path.push(destination);
			destination = destination.parentState;
		}
		
		//Printing out the path
		while(!path.isEmpty()) {
			printBoardState(path.pop().myState);
		}
		
	}
	
	private void solveH2() {
		//Initializing the variables
		PriorityQueue<BoardState> pq = new PriorityQueue<BoardState>();		//Using the priority queue to find the lowest cost node
		
		HashSet<String> finalized = new HashSet<String>();					//Keeping the set of finalized states
		List<BoardState> myBoardList = new ArrayList<BoardState>();
		
		//Adding the source board state to the priority queue
		pq.add(new BoardState(state.myState,0,null));
		
		int ct = 0;
		//While the finalized set doesn't contain the goal state, the priority queue is not empty and the number of expanded nodes is less than the maxStates
		while(!finalized.contains("b12345678") && !pq.isEmpty() && ct < maxStates) {
			//Getting the lowest cost State
			BoardState curr = pq.remove();
			
			if(finalized.contains(curr.myState))
				continue;
			
			//System.out.printf("The state: %s has been finalized!\n",curr.myState);
			//Adding the state to the set of finalized states
			//System.out.printf("The state: %s has been finalized!\n",curr.myState);
			finalized.add(curr.myState);
			myBoardList.add(curr);
			
			//Generating the children of the current BoardState using the generateChildren method of the BoardState class
			List<BoardState> tempList = curr.generateChildrenH2();
			for(int i = 0; i < tempList.size(); i++) {
				BoardState temp = tempList.get(i);
				
				//Adding the child to the priority queue if it hasn't been finalized yet
				if(!finalized.contains(temp.myState)) {
					pq.add(temp);
				}
			}
			
			ct++;
		}
		
		//If we exited because of the Maximum limit being reached
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
		
		//Getting the path by backtracking
		Stack<BoardState> path = new Stack<BoardState>();
		while(destination != null) {
			path.push(destination);
			destination = destination.parentState;
		}
		
		int numMoves = 0;
		//Printing out the path
		while(!path.isEmpty()) {
			printBoardState(path.pop().myState);
			numMoves++;
		}
		
		System.out.printf("We solved the puzzle in %d moves\n",numMoves);
		
	}
	
	
	//Private helper method that evaluates the heuristic h1 for the current state configuration
	private int h1(String myState) {
		int misplaced = 0;
		char[] tiles = {'b','1','2','3','4','5','6','7','8'};
		for(int i = 0; i < 9; i++) {
			//Checking if the nth tile is displaces
			if(myState.charAt(i) != tiles[i])
				misplaced++;
		}
		return misplaced;
	}
	
	//Private helper method that evaluates the heuristic h2 for the current state configuration
	private int h2(String tempState) {
		int distanceSum = 0;
		char temp;
		for(int i = 0; i < 9; i++) {
			temp = tempState.charAt(i);
			//Calculating the Manhattan distance 
			if(temp == 'b') 
				distanceSum += (i-0);
			else
				distanceSum += Math.abs(i-(temp-48));
		}
		//Returns the Manhattan distance
		return distanceSum;
	}
	
	//Helper to move the blank state up
	private BoardState moveUp(BoardState myState) {
		//Checking if we can move up
		if(myState.myBlankTile < 3) {
			return myState;
		}
		
		//Swapping the tile positions using the swap() helper method
		myState = swap(myState.myBlankTile, myState.myBlankTile - 3, myState);
		
		//Changing the position of the blank tile of the BoardState
		myState.myBlankTile -= 3;
		return myState;
	}
	
	//Helper to move the blank state down
	private BoardState moveDown(BoardState myState) {
		//Checking if we can move down
		if(myState.myBlankTile >= 6) {
			return myState;
		}
		
		//Swapping the tile positions using the swap() helper method
		myState = swap(myState.myBlankTile, myState.myBlankTile + 3, myState);
		
		//Changing the position of the blank tile of the BoardState
		myState.myBlankTile += 3;
		return myState;
	}

	//Helper to move the blank state left
	private BoardState moveLeft(BoardState myState) {
		//Checking if we can move left
		if(myState.myBlankTile%3 == 0) {
			return myState;
		}
		
		//Swapping the tile positions using the swap() helper method
		myState = swap(myState.myBlankTile, myState.myBlankTile - 1,myState);
		
		//Changing the position of the blank tile of the BoardState
		myState.myBlankTile--;
		return myState;
	}

	//Helper to move the blank state right
	private BoardState moveRight(BoardState myState) {
		//Checking if we can move up
		if((myState.myBlankTile+1)%3 == 0) {
			return myState;
		}
		
		//Swapping the tile positions using the swap() helper method
		myState = swap(myState.myBlankTile, myState.myBlankTile + 1,myState);
		
		//Changing the position of the blank tile of the BoardState
		myState.myBlankTile++;
		return myState;
	}
	
	
	//Helper method to swap the tiles of a state
	private BoardState swap(int i, int j, BoardState myState) {
		//Getting the string of the state
		String tempState = myState.myState;
		
		//Getting the tiles to swap
		char c1 = tempState.charAt(i);
		char c2 = tempState.charAt(j);
		StringBuilder temp = new StringBuilder();
		
		//Rewriting the tiles of the string
		for(int index = 0; index < 9; index++) {
			//Swapping the tiles
			if(index == i) {
				temp.append(c2);
			} 
			else if(index == j) {
				temp.append(c1);
			}
			
			//Non-swapping tile, we will just add it as it is
			else {
				temp.append(tempState.charAt(index));
			}	
		}
		
		//Returning the new State
		tempState = temp.toString();
		myState.myState = tempState;
		return myState;
	}
	
	
	//Helper method to get the position of the blank tile
	private int getBlankTile() {
		
		//Iterating and checking through the loop
		for(int i = 0; i < 9; i++) {
			if(state.myState.charAt(i) == 'b')
				return i;
		}
		return 0;
	}
	
	//Method to read the commands from a text file
	public void readCommands(String filepath) throws Exception {
		//Creating the buffered reader object
		BufferedReader br = new BufferedReader(new FileReader(new File(filepath)));
		
		String line;
		
		//Checking the commands and using the correct methods based with the correct arguments
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
	
	
	//Helper method that tells us if a string is a valid 8-puzzle state
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
	
	//Printing the board as a 3x3 matrix
	public void printBoard() {
		System.out.printf("%s\n", state.myState.substring(0, 3));
		System.out.printf("%s\n", state.myState.substring(3, 6));
		System.out.printf("%s\n", state.myState.substring(6, 9));
		System.out.println();
	}
	
	//Class to represent a BoardState
	private class BoardState implements Comparable<BoardState>{
		//Instance variables that are necessary for BoardStates
		int functionValue;				//Holds the function value of the current state
		String myState;					//Holds the String configuration of the state
		BoardState parentState;			//Holds the reference to the parent of the state, to get the path
		int myBlankTile;				//Holds the position for the blank tile
		
		//Constructor for the object
		private BoardState(String myState, int functionValue, BoardState parentState) {
			this.functionValue = functionValue;
			this.myState = myState;
			this.parentState = parentState;
			for(int i = 0; i < 9; i++) {
				if(myState.charAt(i) == 'b')
					myBlankTile = i;
			}
		}
		
		//Implementing the compareTo method, so that BoardStates can be added to Priority Queues
		public int compareTo(BoardState o) {
			return this.functionValue - o.functionValue;
		}
		
		//Helper method to create a list of its children for H1()
		/* Logic: Makes all valid moves
		 * Creates a new State for each move (if move is valid)
		 * Sets the function value to depth + h1()
		 * Sets the parent of the generated child to itself
		 * */
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
		
		//Helper method to create a list of its children for H2()
				/* Logic: Makes all valid moves
				 * Creates a new State for each move (if move is valid)
				 * Sets the function value to depth + h2()
				 * Sets the parent of the generated child to itself
				 * */
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
		
		///Helper method to create a list of its children for Local Beam
		/* Logic: Makes all valid moves
		 * Creates a new State for each move (if move is valid)
		 * Sets the function value to h2(), Manhattan distance
		 * Sets the parent of the generated child to itself
		 * */
		public List<BoardState> generateChildrenLocalBeam(){
			List<BoardState> childrenList = new ArrayList<BoardState>();
			BoardState temp = new BoardState(this.myState,this.functionValue,this.parentState);
			
			//Up Child
			temp = moveUp(temp);
			if(!temp.myState.equals(this.myState)) {
				childrenList.add(new BoardState(temp.myState,h2(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			
			//Down Child
			temp = moveDown(temp);
			if(!temp.myState.equals(this.myState)) {
				childrenList.add(new BoardState(temp.myState,h2(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			//Left Child
			temp = moveLeft(temp);
			if(!temp.myState.equals(this.myState)) {
				childrenList.add(new BoardState(temp.myState,h2(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			//Right Child
			temp = moveRight(temp);
			if(!temp.myState.equals(this.myState)) {
				childrenList.add(new BoardState(temp.myState,h2(temp.myState),this));
				temp = new BoardState(this.myState,this.functionValue,this.parentState);
			}
			
			return childrenList;
		}
		
		
	}
	
	//Method to print the Board State as a 3x3 matrix
	private void printBoardState(String tempState) {
		System.out.printf("%s\n", tempState.substring(0, 3));
		System.out.printf("%s\n", tempState.substring(3, 6));
		System.out.printf("%s\n", tempState.substring(6, 9));
		System.out.println();
	}

}
