import pandas as pd
import numpy as np
from decision_tree_id3 import DecisionTreeID3  # Assuming you have saved the decision tree code in 'decision_tree_id3.py'
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logistic_regression import LogisticRegression


def main():
    # Load the data from the CSV file
    data = pd.read_csv('/Users/rohansingh/github_repos/Artificial_Intelligence/AI-ML Source Code/Intro to Machine Learning/Logistic Regression/test.csv')

    # Split the data into features (X) and target labels (y)
    X = data.drop('Target', axis=1).values
    y = data['Target'].values

    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = LogisticRegression()
    regressor.fit(X,y)

    # Adding a new feature which is the predicted values from the Logistic Regression Classifier
    logistic_predictions = regressor.predict(X)

    new_data = []

    i = 0
    for vec in X:
        vec_list = vec.tolist()
        vec_list.append(logistic_predictions[i])
        new_data.append(vec_list)
        i += 1

    #X = np.append(X, logistic_predictions)

    print("\n")

    print("\n")

    new_data_arr = np.array(new_data)


    # Instantiate and train the decision tree
    decision_tree = DecisionTreeID3()
    decision_tree.fit(new_data_arr, y)

    # Example: Make predictions
    test_example = np.array([80, 2, 1])  # Replace with your test data
    prediction = decision_tree.predict([test_example])

    if prediction[0] == 1:
        print("Predicted: Spam")
    else:
        print("Predicted: Not Spam")

    y_pred = decision_tree.predict(new_data_arr)

    


    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy of the decision tree: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
