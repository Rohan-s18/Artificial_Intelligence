"""
Main Running Code for Dominant
"""


# imports
from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse

from dominant import Dominant
from data_preproccessing import load_dataset



# loss function for the final anomaly detection step
def loss(A, A_hat, X, X_hat, alpha):
    # attribute losses
    X_diff = torch.pow(X_hat-X,2)
    X_err = torch.sqrt(torch.sum(X_diff, 1))
    X_cost = torch.mean(X_err)

    # structural losses
    A_diff = torch.pow(A_hat-A,2)
    A_err = torch.sqrt(torch.sum(A_diff, 1))
    A_cost = torch.mean(A_err)

    # total losses
    cost = alpha*X_err + (1-alpha)*A_err
    return cost, A_cost, X_cost


# function to train dominant
def train(filepath, hidden_dim, dropout, lr, alpha, epoch):
    adj, attrs, label, adj_label = load_dataset(filepath=filepath)

    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)

    model = Dominant(feat_size = attrs.size(1), hidden_size = hidden_dim, dropout = dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    # training over the loss function
    for epoch in range(epoch):
        model.train()
        optimizer.zero_grad()
        A_hat, X_hat = model(attrs, adj)
        loss, struct_loss, feat_loss = loss(adj_label, A_hat, attrs, X_hat, alpha)
        l = torch.mean(loss)
        l.backward()
        optimizer.step()        
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))

        if epoch%10 == 0 or epoch == epoch - 1:
            model.eval()
            A_hat, X_hat = model(attrs, adj)
            loss, struct_loss, feat_loss = loss(adj_label, A_hat, attrs, X_hat, alpha)
            score = loss.numpy()
            print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(label, score))



# main function to run the code
def main():

    train(filepath="/Users/rohansingh/github_repos/Artificial_Intelligence/Graph ML/GCN-based-GAE/DOMINANT/dataset/test.mat", 
          hidden_dim=64, dropout=0.3, lr=5e-3,alpha=0.8,epoch=50)


if __name__ == '__main__':
    main()
