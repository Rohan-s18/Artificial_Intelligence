"""
Author: Rohan Singh
Utility Code and functions for Dtree
"""

import numpy as np

def calculate_entropy(y):
    """
    Calculate the entropy of a set of labels.
    """
    unique_labels, label_counts = np.unique(y, return_counts=True)
    probabilities = label_counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_information_gain(X, y, feature_idx):
    """
    Calculate the information gain for a feature in the dataset.
    """
    total_entropy = calculate_entropy(y)
    unique_values, value_counts = np.unique(X[:, feature_idx], return_counts=True)
    weighted_entropy = 0
    for value, count in zip(unique_values, value_counts):
        subset_indices = X[:, feature_idx] == value
        subset_entropy = calculate_entropy(y[subset_indices])
        weighted_entropy += (count / len(y)) * subset_entropy
    information_gain = total_entropy - weighted_entropy
    return information_gain

def calculate_split_entropy(X, y, feature_idx):
    """
    Calculate the split entropy for a feature in the dataset.
    """
    unique_values, value_counts = np.unique(X[:, feature_idx], return_counts=True)
    split_entropy = 0
    for value, count in zip(unique_values, value_counts):
        subset_indices = X[:, feature_idx] == value
        subset_probability = count / len(y)
        split_entropy -= subset_probability * np.log2(subset_probability)
    return split_entropy

def calculate_gain_ratio(X, y, feature_idx):
    """
    Calculate the gain ratio for a feature in the dataset.
    """
    information_gain = calculate_information_gain(X, y, feature_idx)
    split_entropy = calculate_split_entropy(X, y, feature_idx)
    
    # Avoid division by zero
    if split_entropy == 0:
        return 0
    
    gain_ratio = information_gain / split_entropy
    return gain_ratio

def split_data(X, y, feature_idx, value):
    """
    Split the data into subsets based on a feature and its value.
    """
    subset_indices = X[:, feature_idx] == value
    subset_X = X[subset_indices]
    subset_y = y[subset_indices]
    return subset_X, subset_y

def majority_vote(y):
    """
    Return the majority class label in a set of labels.
    """
    unique_labels, label_counts = np.unique(y, return_counts=True)
    majority_label = unique_labels[np.argmax(label_counts)]
    return majority_label

