"""
Author: Rohan Singh
Python module to clean text from html to regular text
Date: 28 May 2023
"""

#  Imports
import pandas as pd
import re
import string
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from tqdm import tqdm

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import random

from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

#stop=set(stopwords.words('english'))



#  Helper function to remove html tags
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


#  Main function for testing
def main():
    example = """<div>
<h1>Real or Fake</h1>
<p>Kaggle </p>
<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>
</div>"""

    print(remove_html(example))


if __name__ == "__main__":
    main()


