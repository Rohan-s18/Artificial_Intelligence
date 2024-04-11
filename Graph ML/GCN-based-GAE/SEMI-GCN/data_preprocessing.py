"""
python script to pre-process the data
"""


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import time
import math
import random






manual_seed=7


def load_data(filepath):
    # loading node information
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names =  feature_names + ["subject"]
    node_data = pd.read_csv(filepath + 'cora.content', sep='\t', names=column_names)
    edge_list = pd.read_csv(filepath+ 'cora.cites', sep='\t', names=['target', 'source'])

    num_node, num_feature = node_data.shape[0], node_data.shape[1]-1
    node_index = np.array(node_data.index)



    # labelling anomaly instances
    node_data['label'] = node_data['subject'].apply(lambda x: 'anomaly' if x == 'Rule_Learning' else 'normal')
    node_data['label'].value_counts()

    indices = np.arange(num_node)
    unlabeled_data, labeled_data, index_unlabeled, index_labeled = train_test_split(node_data, indices, test_size=0.1, stratify=node_data['label'], random_state=123)

    # set label of instances to 'unknown' as unlabeled instances
    for idx in index_unlabeled:
        node_data['label'][node_index[idx]] = 'unknown'


    # performing a test-train split on the data
    indices = np.arange(num_node)
    train_data, test_data, index_train, index_test = train_test_split(node_data, indices, test_size=0.2, stratify=node_data['label'], random_state=manual_seed)
    train_data, val_data, index_train, index_val = train_test_split(train_data, index_train, test_size=0.2, stratify=train_data['label'], random_state=manual_seed)

    index_normal_train = []
    index_anomaly_train = []
    for i in index_train:
        if node_data.iloc[i]['label'] == 'normal':
            index_normal_train.append(i)
        elif node_data.iloc[i]['label'] == 'anomaly':
            index_anomaly_train.append(i)

    index_normal_val = []
    index_anomaly_val = []
    for i in index_val:
        if node_data.iloc[i]['label'] == 'normal':
            index_normal_val.append(i)
        elif node_data.iloc[i]['label'] == 'anomaly':
            index_anomaly_val.append(i)
        
    index_normal_test = []
    index_anomaly_test = []
    for i in index_test:
        if node_data.iloc[i]['label'] == 'normal':
            index_normal_test.append(i)
        elif node_data.iloc[i]['label'] == 'anomaly':
            index_anomaly_test.append(i)
        

    # Categorizing classes
    def f(x):
        if x == "anomaly":
            return 1
        elif x == "normal":
            return 0
        else:
            return -1

    labels = node_data['label']
    cat_labels = labels.apply(f)

    # creating torch tensors
    cat_labels = torch.LongTensor(np.array(cat_labels))
    features = torch.FloatTensor(np.array(node_data[feature_names]))

    index_normal_train = torch.LongTensor(index_normal_train)
    index_anomaly_train = torch.LongTensor(index_anomaly_train)

    index_normal_val = torch.LongTensor(index_normal_val)
    index_anomaly_val = torch.LongTensor(index_anomaly_val)

    index_normal_test = torch.LongTensor(index_normal_test)
    index_anomaly_test = torch.LongTensor(index_anomaly_test)


    # obtaining the adjacency matrix
    index_map = {j: i for i, j in enumerate(node_index)}
    adj = np.zeros((num_node, num_node))


    for i in range(edge_list.shape[0]):
        u = edge_list['target'][i]
        v = edge_list['source'][i]
        adj[index_map[u], index_map[v]] = 1
        adj[index_map[v], index_map[u]] = 1


    I = np.eye(num_node)
    adj_tld = adj + I

    # symmetric normalization
    rowsum = np.sum(adj_tld, axis=1)
    r_inv = rowsum ** -0.5
    r_inv[np.isinf(r_inv)] = 0.    # check devided by 0
    r_mat_inv = np.diag(r_inv)

    adj_hat = np.dot( np.dot(r_mat_inv, adj_tld), r_mat_inv)    # r_mat_inv * adj_tld * r_mat_inv

    adj = torch.FloatTensor(np.array(adj_hat))

    return features, adj, index_anomaly_train, index_normal_train, index_anomaly_val, index_normal_val, index_anomaly_test, index_normal_test, cat_labels

