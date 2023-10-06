import numpy as np
from collections import Counter


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from logistic_regression import LogisticRegression

class Node:
    def __init__(self, data):
        self.data = data
        self.children = {}
        self.label = None

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, splits):
    total_entropy = entropy(y)
    weighted_entropy = sum((len(subset) / len(y)) * entropy(subset) for subset in splits)
    return total_entropy - weighted_entropy

def split_data(X, y, feature_index):
    unique_values = np.unique(X[:, feature_index])
    splits = [([], []) for _ in unique_values]

    for i, value in enumerate(X[:, feature_index]):
        index = np.where(unique_values == value)[0][0]
        splits[index][0].append(X[i])
        splits[index][1].append(y[i])

    return splits

def build_tree(X, y, features):
    if len(np.unique(y)) == 1:
        leaf = Node(data=None)
        leaf.label = y[0]
        return leaf

    if len(features) == 0:
        leaf = Node(data=None)
        leaf.label = Counter(y).most_common(1)[0][0]
        return leaf

    best_feature_index = None
    best_information_gain = -1

    for feature_index in features:
        splits = split_data(X, y, feature_index)
        gain = information_gain(y, [split[1] for split in splits])

        if gain > best_information_gain:
            best_feature_index = feature_index
            best_information_gain = gain

    if best_information_gain == 0:
        leaf = Node(data=None)
        leaf.label = Counter(y).most_common(1)[0][0]
        return leaf

    remaining_features = [f for f in features if f != best_feature_index]
    root = Node(data=best_feature_index)
    splits = split_data(X, y, best_feature_index)

    for i, subset in enumerate(splits):
        child = build_tree(np.array(subset[0]), np.array(subset[1]), remaining_features)
        root.children[i] = child

    return root


def predict_tree(node, X):
    if node.label is not None:
        return node.label

    feature_index = node.data
    value = X[feature_index]

    if value in node.children:
        return predict_tree(node.children[value], X)
    else:
        return None

class DecisionTreeID3:
    def __init__(self):
        self.root = None

        self.max_label = None

        # Regressor Object for the tree
        self.regressor = None

    def fit(self, X, y):

        """
        self.regressor = LogisticRegression()
        self.regressor.fit(X,y)

        # Adding a new feature which is the predicted values from the Logistic Regression Classifier
        logistic_predictions = self.regressor.predict(X)


        X = np.append(X, logistic_predictions)
        """

        self.max_label = np.argmax(np.bincount(y))

        num_features = X.shape[1]
        features = list(range(num_features))
        self.root = build_tree(X, y, features)

    def predict(self, X):

        """
        logistic_predictions = self.regressor.predict(X)
        X = np.append(X, logistic_predictions)
        """

        predictions = []
        for x in X:
            prediction = predict_tree(self.root, x)
            if prediction == None:
                predictions.append(self.max_label)
            else:
                predictions.append(prediction)
        return predictions


def main():
    iris_data = load_iris()

    data = np.array(iris_data.data)
    target = np.array(iris_data.target)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    regressor = DecisionTreeID3()
    regressor.fit(X_train,y_train)

    print("\n")
    print(accuracy_score(regressor.predict(X_test), y_test))
    print("\n")

if __name__ == "__main__":
    main()