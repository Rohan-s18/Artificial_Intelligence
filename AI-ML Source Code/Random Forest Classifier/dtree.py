"""
Author: Rohan Singh
DTree code
"""

import numpy as np
from util import calculate_entropy, calculate_information_gain, majority_vote

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            # If we reach the maximum depth or pure leaf node, return the majority class
            return majority_vote(y)
        
        if len(X) == 0:
            # If there is no data, return the majority class
            return majority_vote(y)
        
        best_feature = self.choose_best_feature(X, y)
        if best_feature is None:
            return majority_vote(y)
        
        tree = {"feature": best_feature, "children": {}}
        
        unique_values = np.unique(X[:, best_feature])
        for value in unique_values:
            subset_X, subset_y = self.split_data(X, y, best_feature, value)
            tree["children"][value] = self.fit(subset_X, subset_y, depth + 1)
        
        self.tree = tree
        return tree

    def predict(self, X):
        if self.tree is None:
            raise ValueError("The tree is not trained.")
        predictions = [self.predict_single(x, self.tree) for x in X]
        return predictions

    def predict_single(self, x, node):
        if "feature" in node:
            feature = node["feature"]
            value = x[feature]
            if value in node["children"]:
                return self.predict_single(x, node["children"][value])
            else:
                # If the feature value is not in the tree, return the majority class
                return majority_vote(list(node["children"].values()))
        else:
            return node

    def choose_best_feature(self, X, y):
        num_features = X.shape[1]
        best_feature = None
        best_info_gain = -1
        for feature_idx in range(num_features):
            info_gain = calculate_information_gain(X, y, feature_idx)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature_idx
        return best_feature

    def split_data(self, X, y, feature_idx, value):
        subset_indices = X[:, feature_idx] == value
        subset_X = X[subset_indices]
        subset_y = y[subset_indices]
        return subset_X, subset_y
