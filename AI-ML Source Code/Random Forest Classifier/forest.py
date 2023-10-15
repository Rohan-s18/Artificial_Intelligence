"""
Author: Rohan Singh
Random Forest Classifier Code
"""


import numpy as np
from dtree import DecisionTree
from util import bootstrap_sample

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Create a bootstrap sample of the data
            bootstrap_X, bootstrap_y = bootstrap_sample(X, y)
            
            # Train a decision tree on the bootstrap sample
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(bootstrap_X, bootstrap_y)
            
            self.trees.append(tree)

    def predict(self, X):
        if not self.trees:
            raise ValueError("The forest is not trained.")
        
        # Make predictions from each tree and perform majority voting
        predictions = np.array([tree.predict(X) for tree in self.trees])
        majority_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_predictions
