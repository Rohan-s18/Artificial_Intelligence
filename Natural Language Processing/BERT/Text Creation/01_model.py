"""
Author: Rohan Singh
Date: August 3, 2023
This python script contains code for free text generation using BERT and hill-climb search
"""


# Imports
import numpy as np

#import tensorflow as tf
#import torch

#from transformers import BertForMaskedLM, BertTokenizerFast
from transformers import pipeline


# Function to return the k tokens (with the highest probablity) after prompt
def produce_text(pipe, prompt, k):
    pass


# Main function
def main():
    
    model = pipeline('fill-mask', model='bert-base-uncased')

    print(model('istanbul is a great [MASK]'))
    


if __name__ == "__main__":
    main()