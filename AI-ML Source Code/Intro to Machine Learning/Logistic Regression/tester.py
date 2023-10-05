import pandas as pd
import numpy as np
from decision_tree_id3 import DecisionTreeID3  # Assuming you have saved the decision tree code in 'decision_tree_id3.py'

def main():
    # Load the data from the CSV file
    data = pd.read_csv('/Users/rohansingh/github_repos/Artificial_Intelligence/AI-ML Source Code/Intro to Machine Learning/Logistic Regression/test.csv')

    # Split the data into features (X) and target labels (y)
    X = data.drop('Target', axis=1).values
    y = data['Target'].values

    # Instantiate and train the decision tree
    decision_tree = DecisionTreeID3()
    decision_tree.fit(X, y)

    # Example: Make predictions
    test_example = np.array([80, 2, 1])  # Replace with your test data
    prediction = decision_tree.predict([test_example])

    if prediction[0] == 1:
        print("Predicted: Spam")
    else:
        print("Predicted: Not Spam")

if __name__ == "__main__":
    main()
