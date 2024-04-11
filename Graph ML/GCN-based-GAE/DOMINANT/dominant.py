"""
Python Source code for "Deep Anomaly Detection on Attributed Networks" (Dominant)
Paper Source: https://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf
Author: Rohan Singh
This is the main model code for the paper
"""


# imports
import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution



# class for the attributed network encoder
class Encoder(nn.Module):

    # initialization function
    def __init__(self, num_features, num_hidden_layers, dropout) -> None:
        super(Encoder, self).__init__()
        self.gcn_1 = GraphConvolution(num_features, num_hidden_layers)
        self.gcn_2 = GraphConvolution(num_hidden_layers, num_hidden_layers)
        self.dropout = dropout


    # forward pass function
    def forward(self, X, A):
        X = F.relu(self.gcn_1(X,A))
        X = F.dropout(X,self.dropout,training=self.training)
        X = F.relu(self.gcn_2(X,A))

        return X



# class for the attribute decoder 
class AttributeDecoder(nn.Module):

    # initialization function
    def __init__(self, num_features, num_hidden_layers, dropout) -> None:
        super(AttributeDecoder, self).__init__()
        self.gcn_1 = GraphConvolution(num_features, num_hidden_layers)
        self.gcn_2 = GraphConvolution(num_hidden_layers, num_hidden_layers)
        self.dropout = dropout


    # forward pass function
    def forward(self, X, A):
        X = F.relu(self.gcn_1(X,A))
        X = F.dropout(X,self.dropout,training=self.training)
        X = F.relu(self.gcn_2(X,A))

        return X

    

# class for the structure decoder
class StructureDecoder(nn.Module):

    # initialization function
    def __init__(self, num_hidden_layers, dropout) -> None:
        super(StructureDecoder, self).__init__()
        self.gcn = GraphConvolution(num_hidden_layers, num_hidden_layers)
        self.dropout = dropout

    # forward pass function
    def forward(self, X, A):
        X = F.relu(self.gcn(X,A))
        X = F.dropout(X, self.dropout, training=self.training)
        X = X @ X.T

        return X
    


# class for dominant, this is the proposed model
class Dominant(nn.Module):

    # initialization function
    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant, self).__init__()
        
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = AttributeDecoder(feat_size, hidden_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)
    

    # forward pass function
    def forward(self, x, adj):
        x = self.shared_encoder(x, adj)
        x_hat = self.attr_decoder(x, adj)
        struct_reconstructed = self.struct_decoder(x, adj)
        return struct_reconstructed, x_hat
    