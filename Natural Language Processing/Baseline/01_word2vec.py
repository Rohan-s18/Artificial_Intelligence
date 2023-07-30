# Data engineering libraries
import numpy as np
import pandas as pd

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Spacy libraries
import en_core_web_sm
import spacy
from spacy.lang.en import English