# Artificial Intelligence
## Author: Rohan Singh

This repository is for my personal AI projects. The source code is in both **Python** and **Java**, the code in both of these directories can be used as a foundation for other AI programs that implement these algorithms. All of the source code in this directory is **made from scratch and doesn't use any external libraries like Scikit-Learn or Prophet.**

Other than the customizable templates for AI, there are also some projects that integrate these AI algorithms with real-applications, mostly games that are solved using AI algorithms.  

This repository is divided into 4 different parts:   
  - **AI/ML Source Code:** This contains the source code for different AI/ML algorithms in Java and Python.  
  - **Neural Networks:** This contains an exstensive collection of general Neural Networks in Python.  
  - **High Performance ML:** This contains C++ source code for High-Performance Machine Learning.  
  - **AI based implementations:** This contains some implementations of the AI/ML algorithms that I developed.  
  - **AI Help:** This contains source code for the prerequisites of AI, such as probability, graph ADTs and Python.  

## AI/ML Source Code:   
This Directory contains the source code for different AI/ML algorithms which are further divided into their own subgroups within this directory. These subgroups include:   
1) **Problem Solving with Search: (Finished)**  
    - **Uninformed Search**: This includes depth-first, breadth-first traversals and uniform-cost search (Also in Python).  
    - **Informed Search:** This includes A* algorithm.    
    - **Search for Optimization:** This includes Hill Climb Search, Simulated Annealing and beam with k states.   
    
2) **Machine Learning: (Finished)**    
    - **Clustering:**  Classifcation of data using KMeans clustering (unsupervised).  
    - **Linear decision boundaries using a neural network:** Classification of data using Logistic Regression.    
    - **Neural Network Error:** Calculating error and gradient for a given neural network.  
    - **Neural Network Optimization:** Updating Neural network decision boundares using gradient descent optimization.   
  
3) **Decision Making under uncertainity and NLP:** *Not implemented yet*  
    - Sequential Decision Making  
    - Reinforcement Learning  
    - Natural Language Processing    

## Neural Networks
This Directory contains the source code for different types of General Neural Networks developed from scratch. These Neural Networks are divided into theior own subcategories, and can be used based in different applications since they are more general.  
The types of Neural Networks that are included here are:  
  - **Single Layer (Multi-input -> Single-output):** This directory contains source code for a general single layer n-to-1 neural network.   
  - **Single Layer (Multi-input -> Multi-output):** This directory contains source code for a general single layer n-to-m neural network.  
  - **Hidden Layer (Multi-input -> Single-output):** This directory contains source code for a general non-linearity (hidden layer) n-to-1 neural network.      
  - **Hidden Layer (Multi-input -> Multi-output):** This directory contains source code for a general non-linearity (hidden layer) n-to-m neural network. 

## High-Performance ML
This directory contains the source code for High-Performace machine learning code in C++. Since it is compiled code, it is much faster than interpreted (Python) ML code.  
Machine Learning Code:  
  - **Linear Regression:** Single Variable Linear Regression has been implemented.  
  - **KMeans Clustering:** Currently in progress.  

## AI-based Implementations:  
1) **Eight puzzle solver:** Source Code for an *Eight-Puzzle* solving algorithm. The source code includes both the game (which someone could play) as well as the solving algorithm. This was achieved using both A* search algorithm as well as k-states beam search. Source code is in Java.    
2) **Iris species classifier:** Source Code for classifying the *species of Iris flowers* given their petal/sepal dimensions. The Source Code is in Python and it includes:  
    - **Unsupervised Learning:** Used KMeans clustering to find the Iris species.  
    - **Neural Network:** Developoed a Neural Network to classify the species of Iris using logistic regression and optimized the neural network weights using gradient descent.  
    - **Multi Layer Perceptron:** This is not home-made code, but rather using Scikit-learn's Multi-Layer Perceptron to classify the different iris classes.  
    
## AI Help Directories:
1) **AI-prereq:** The AI-prereq directory doesn't contain any AI-based alorithms, but just the basic stuff that people should know before they dive into the rest of the code. This includes a basics of python (pyplot) as well as Graph data structures. If you have experience in Python and Data Structures/Algorithms, you won't face any problem here.  
2) **Probability Modules:** This directory can be used to refresh yourself with Probability and includes python source code that implement different things in probability, like Bernoulli's Trials, Probabilstic Classification and Bayesian Learning.   
3) **Python:** This includes python code for vector calculus and linear algebra that you may need for different machine learning algorithms.  
    

You can contact me at:  
rohan.b.singh54@gmail.com  
rxs1182@case.edu  
