"""
Author: Rohan Singh
Date: July 25, 2023
This module contains code for a simple sequence classification model using DistilBERT
"""

#%%
"""
Librarires Used
"""

# EDA Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score

# BERT Support Libraries
import torch
import tensorflow as tf
import en_core_web_sm
import spacy
from spacy.lang.en import English
from datasets import Dataset, load_metric

# BERT Libraries
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Pipeline, Trainer, TrainingArguments

#%%
"""
Function for making a dataset
"""

def make_dataset(df, text_title, sequence_title, split_size):
    pass


#%%
"""
Function to preprocess the Data
"""

def preprocess_function(examples):
    pass


#%%
"""
Function to tokenize the dataset
"""

def tokenize_dataset(dataset, tokenizer, stride):
    pass


#%%
"""
Main function to run the modek
"""

def main():
    pass

if __name__ == "__main__":
    main()


#%%