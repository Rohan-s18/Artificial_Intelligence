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
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline, Trainer, TrainingArguments, DataCollatorWithPadding


#%%
"""
This cell contains constants
"""

STRIDE=32

nlp = spacy.load('en_core_web_sm')



#%%
"""
Function for making a dataset
"""

def make_dataset(df, text_title, sequence_title, split_size):
    df = df[df[text_title].str.len() < 500]

    return Dataset.from_dict(
        dict(
            utterances=df[text_title].to_list(),
            tokenized_utterances=df[text_title].str.split(),
            labels=df[sequence_title].to_list()
        )
    ).train_test_split(test_size=split_size)


#%%
"""
Function to preprocess the Data
"""

def preprocess_function(tokenizer, examples):
    return tokenizer(examples["utterance"],truncation=True, stride=STRIDE)


#%%
"""
Function to tokenize the dataset
"""

def tokenize_dataset(dataset, tokenizer):
    return dataset.map(preprocess_function,batched=True)


#%%
"""
Main function to run the modek
"""

def main():
    pass

if __name__ == "__main__":
    main()


#%%