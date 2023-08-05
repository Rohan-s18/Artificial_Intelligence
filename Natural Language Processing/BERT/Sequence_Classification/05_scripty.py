"""
Author: Rachel Tjarksen
Date: August 3, 2023
Script for a DistilBERT model for sequence classification
"""

#%%
"""
This cell contains the libraries used in our script
"""

# EDA Libraries
import numpy as np
import pandas as pd

# Scikit learn libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

# BERT support libraries
import torch
import tensorflow as tf
from datasets import Dataset, load_metric

# BERT libraries
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, DataCollatorWithPadding, pipeline, Trainer, TrainingArguments 


#%%
"""
This cell contains code for data preprocessing
"""

df = pd.read_csv("/Users/racheltj/Desktop/java/xortorch/modelling/general_dataset.csv")

utterances = df["Title"].to_list()
tokenized_utterances = df["Title"].str.split().to_list()
sequence_labels = []

for color in df["Color"]:
    if color == "G":
        sequence_labels.append(0)
    else:
        sequence_labels.append(1)


#%%
"""
This cell contains code for Dataset Creation
"""

title_dataset = Dataset.from_dict(
    dict(
        utterances=utterances,
        tokenized_utterances=tokenized_utterances,
        sequence_labels=sequence_labels
    )
)

title_dataset = title_dataset.train_test_split()

#%%
"""
This cell contains code for Tokenization
"""

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def preprocess_function(examples):
    return tokenizer(examples['utterances'], truncation=True, stride=4)

tokenized_dataset = title_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#%%
"""
This cell contains the code for Model Setup
"""
id2label = {0:"POSITIVE",1:"NEGATIVE"}
label2id = {"POSITIVE":0,"NEGATIVE":1}

seq_clf_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2, id2label=id2label, label2id=label2id)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

epochs = 20

training_args = TrainingArguments(
    output_dir="../models/titles_clf",
    num_train_epochs=epochs,
    per_device_train_batch_size=32,
    per_gpu_eval_batch_size=32,
    load_best_model_at_end=True,

    warmup_steps=len(tokenized_dataset['train']),
    weight_decay=0.05,

    logging_steps=1,
    log_level='info',
    evaluation_strategy='epoch',
    save_strategy='epoch'

)

trainer = Trainer(
    model=seq_clf_model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics,
    data_collator=data_collator
)



#%%
"""
This cell contains code for training the model
"""
trainer.evaluate()

print("\n\n")

trainer.train()

print("\n\n")

trainer.evaluate()


#%%