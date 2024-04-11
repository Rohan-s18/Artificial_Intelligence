"""
Semi-supervised Anomaly Detection on Attributed Graphs implementation for CSDS 491
Author: Rohan Singh
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
from data_preprocessing import load_data




manual_seed = 1




# creating the gcn class
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout = dropout
        
        # weight and bias between input and hidden layer
        self.weight_in_hid = Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias_in_hid = Parameter(torch.FloatTensor(nhid))
        
        # weight and bias between hidden and output layer
        self.weight_hid_out = Parameter(torch.FloatTensor(nhid, nclass))
        self.bias_hid_out = Parameter(torch.FloatTensor(nclass))
        
        self.drop_layer = nn.Dropout(p=self.dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.manual_seed(manual_seed)
        stdv_in_hid = math.sqrt(6) / math.sqrt(self.nfeat + self.weight_in_hid.size(1))
        self.weight_in_hid.data.uniform_(-stdv_in_hid, stdv_in_hid)
        self.bias_in_hid.data.uniform_(-stdv_in_hid, stdv_in_hid)
        
        stdv_hid_out = math.sqrt(6) / math.sqrt(self.weight_in_hid.size(1) + self.weight_hid_out.size(1))
        self.weight_hid_out.data.uniform_(-stdv_hid_out, stdv_hid_out)
        self.bias_hid_out.data.uniform_(-stdv_hid_out, stdv_hid_out)
        
        
    def forward(self, x, adj):
        o_hidden = torch.mm(x, self.weight_in_hid)
        o_hidden = torch.spmm(adj, o_hidden)
        o_hidden = o_hidden + self.bias_in_hid
        o_hidden = F.relu(o_hidden)
        o_hidden = self.drop_layer(o_hidden)
        o_out = torch.mm(o_hidden, self.weight_hid_out)
        o_out = torch.spmm(adj, o_out)
        o_out = o_out + self.bias_hid_out
        return o_out
    



# desigining the early stopping mechanism
class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss




# defning the anomaly scores as outlined in the paper
def anomaly_score(node_embedding, c):
    return torch.sum((node_embedding - c) ** 2)


# defining the normal loss
def nor_loss(node_embedding_list, c):
    s = 0
    num_node = node_embedding_list.size()[0]
    for i in range(num_node):
        s = s + anomaly_score(node_embedding_list[i], c)
    return s/num_node


# defining the AUC loss
def AUC_loss(anomaly_node_emb, normal_node_emb, c):
    s = 0
    num_anomaly_node = anomaly_node_emb.size()[0]
    num_normal_node = normal_node_emb.size()[0]
    for i in range(num_anomaly_node):
        for j in range(num_normal_node):
            s1 = anomaly_score(anomaly_node_emb[i], c)
            s2 = anomaly_score(normal_node_emb[j], c)
            s = s + torch.sigmoid(s1 - s2)
    return s/(num_anomaly_node * num_normal_node) # check devide by zero


# defining the total objective function loss
def objecttive_loss(anomaly_node_emb, normal_node_emb, c, regularizer=1):
    Nloss = nor_loss(normal_node_emb, c)
    AUCloss = AUC_loss(anomaly_node_emb, normal_node_emb, c)
    loss = Nloss - regularizer * AUCloss
    return loss



#  helper function to evaluate the model
def evaluate(model, index_anomaly_test, index_normal_test, cat_labels, features, adj, center):  
    anomaly_labels = list(cat_labels[index_anomaly_test].numpy())
    normal_labels = list(cat_labels[index_normal_test].numpy())

    model.eval()
    
    # anomaly score of instances
    output = torch.FloatTensor(model(features, adj).detach().numpy())
    anomaly = output[index_anomaly_test]
    normal = output[index_normal_test]
    anomaly_s = []
    normal_s = []

    for i in range(anomaly.size()[0]):
        anomaly_s.append(anomaly_score(anomaly[i], center).item())
    
    for i in range(normal.size()[0]):
        normal_s.append(anomaly_score(normal[i], center).item())


    labels = anomaly_labels + normal_labels
    scores = anomaly_s + normal_s

    # calculate roc_auc_score
    auc = roc_auc_score(labels, scores)
    print("AUC score on test set = {}".format(auc))

    # plot roc curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    #plt.plot(fpr, tpr, marker='.', label='GCN')
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()



# function to train the model
def train(features, adj, index_anomaly_train, index_normal_train, index_anomaly_val, index_normal_val, index_anomaly_test, index_normal_test, cat_labels):
    
    # instantiating the model
    model = GCN(nfeat=features.shape[1],
            nhid=32,
            nclass=32,
            dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.eval()
    normal_embed = model(features, adj)[index_normal_train].detach().numpy()
    normal_embed = torch.FloatTensor(normal_embed)
    center = torch.mean(normal_embed,0)
    
    # training the model
    EPOCH = 500
    train_losses = []
    val_losses = []
    AUC_regularizer = 1
    early_stopping = EarlyStopping(patience=20, verbose=True)

    for epoch in range(EPOCH):
        model.train()
        optimizer.zero_grad() 
        output = model(features, adj) 
        loss_train = objecttive_loss(output[index_anomaly_train], output[index_normal_train], center, AUC_regularizer) 
        loss_train.backward() 
        optimizer.step() 
    
        # evaluate in val set
        model.eval()
        output = model(features, adj)
        loss_val = objecttive_loss(output[index_anomaly_val], output[index_normal_val], center, AUC_regularizer) #
    
        train_losses.append(loss_train.detach().item())
        val_losses.append(loss_val.detach().item())
    
        print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()))
    
        early_stopping(loss_val, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    
    # evaluating the model
    evaluate(model=model, index_anomaly_test=index_anomaly_test, index_normal_test=index_normal_test, 
             cat_labels=cat_labels, features=features, adj=adj, center=center)

    # loading the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))


    
# main function to run the code
def main():

    filepath = "/Users/rohansingh/github_repos/Artificial_Intelligence/Graph ML/GCN-based-GAE/SEMI-GCN/cora/"
    features, adj, index_anomaly_train, index_normal_train, index_anomaly_val, index_normal_val, index_anomaly_test, index_normal_test, cat_labels = load_data(filepath=filepath)

    train(features= features,adj=adj, index_anomaly_train= index_anomaly_train, index_normal_train= index_normal_train, 
          index_anomaly_val= index_anomaly_val, index_normal_val= index_normal_val, index_anomaly_test= index_anomaly_test, 
          index_normal_test= index_normal_test, cat_labels= cat_labels)



if __name__ == "__main__":
    main()