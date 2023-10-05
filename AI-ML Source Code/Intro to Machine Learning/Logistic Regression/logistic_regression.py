"""
This Python module contains code for a simple logistic classifier
"""


# Imports
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return [1 if p >= 0.5 else 0 for p in predictions]
    


# Main function for testing
def main():
    iris_data = load_iris()

    data = np.array(iris_data.data)
    target = np.array(iris_data.target)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)



    regressor = LogisticRegression()
    regressor.fit(X_train,y_train)

    print("\n")
    print(accuracy_score(regressor.predict(X_test), y_test))
    print("\n")


if __name__ == "__main__":
    main()
