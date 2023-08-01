"""
Author: Rohan Singh
Date: July 31, 2023
This python module contains code for the demo of my default sequence classifier python module
"""

#%%
"""
Import used
"""

from default_seq_clf import run_default
import pandas as pd


#%%
"""
Main function for demonstration
"""
def main():

    filepath = "/Users/rohansingh/Desktop/df_main.csv"
    df = pd.read_csv(filepath)
    text_title = "Title"
    sequence_title = "Color"
    epochs = 20
    num_labels = 3
    label2id = {"G":0,"Y":1,"R":2}
    id2label = {0:"G",1:"Y",2:"R"}
    output_dir = "/Users/rohansingh/Desktop/model"

    # Getting the model
    model = run_default(df, text_title, sequence_title, epochs, num_labels, label2id, id2label, output_dir)


if __name__ == "__main__":
    main()


#%%