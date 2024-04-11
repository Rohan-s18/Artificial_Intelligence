"""
Data preprocessing Script
"""

# imports
import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random


# function to load the dataset and return A and X as specified in the literature
def load_dataset(filepath):
    
    data_mat = sio.loadmat(filepath)
    adj = data_mat['Network']
    feat = data_mat['Attributes']
    target = data_mat['Label']
    target = target.flatten()

    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.toarray()
    feat = feat.toarray()
    return adj_norm, feat, target, adj


# function to normalize the dataset as specified in the literature
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()