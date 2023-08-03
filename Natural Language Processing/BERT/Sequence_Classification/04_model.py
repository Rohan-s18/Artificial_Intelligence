"""
Author: Rohan Singh and Rachel Tjarksen
Date: August 2, 2023
Sequence Classification Model using DistilBERT
"""

# Imports

# EDA Libraries
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# BERT Support Libraries
#import tensorflow as tf
import torch
from datasets import Dataset, load_metric

# BERT imports
from transformers import TrainingArguments, Trainer, DistilBertTokenizerFast, DistilBertForSequenceClassification, DataCollatorWithPadding, pipeline


# Data pre-process function
def preprocess_function(examples):
    pass


# Main function
def main():
    
    df = pd.read_csv("/Users/rohansingh/Desktop/df_main.csv")


#%%

df = pd.read_csv("/Users/rohansingh/Desktop/df_main.csv")

#%%


if __name__ == "__main__":
    main()


