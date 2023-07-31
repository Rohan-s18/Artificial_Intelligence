"""
Author: Rohan Singh
Date: July 31, 2023
This python module contains code for a defulat sequence classification model using BERT
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

metric = load_metric('accuracy')


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
Function for the computation metric for the training loop
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)




#%%
"""
This function will run the model with the default settings
"""

def run_default(df, text_title, sequence_title, epochs, num_labels, label2id, id2label, output_dir):
    
    # Creating the dataset
    demo_dataset = make_dataset(df, sequence_title=sequence_title, text_title=text_title)

    # Tokenizing the dataset
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    demo_dataset = tokenize_dataset(dataset=demo_dataset)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Loading the model
    sequence_clf_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=num_labels,
                                                                             id2label=id2label,label2id=label2id)
    
    # Setting up the training arguments
    epochs = 20
    training_args = TrainingArguments(

        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_gpu_eval_batch_size=32,
        load_best_model_at_end=True,

        warmup_steps=len(demo_dataset['train']),
        weight_decay=0.05,

        logging_steps=1,
        log_level='info',
        evaluation_strategy='epoch',
        save_strategy='epoch'

    )

    # Setting up the trainer
    trainer = Trainer(
        model=sequence_clf_model,
        args=training_args,
        train_dataset=demo_dataset['train'],
        eval_dataset=demo_dataset['test'],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    # Training the model
    print("\nInitial:\n",trainer.evaluate(),"\n\n")
    trainer.train()
    print("Final:\n",trainer.evaluate(),"\n\n")

    # returning the trained model
    return sequence_clf_model



#%%