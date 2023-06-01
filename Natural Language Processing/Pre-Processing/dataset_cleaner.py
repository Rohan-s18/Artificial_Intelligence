"""
Author: Rohan Singh
Python module to clean html datasets
Date: 28 May 2023
"""


#  Imports
from text_cleaner import remove_html
import pandas as pd
import os


#  Helper function to obtain the messy text from the dataset
def get_messy(html_directory):
    html_list = []

    # Iterate over HTML files in the directory
    for filename in os.listdir(html_directory):
        if filename.endswith(".html"):
            file_path = os.path.join(html_directory, filename)

            with open(file_path, "r") as file:
                html_content = file.read()
                html_list.append(html_content)

    print(len(html_list))
    return html_list

    

#  Helper function to clean all of the data
def clean_all(data):
    cleaned = []

    for i in range(0,len(data), 1):
        cleaned.append(remove_html(data[i]))

    return cleaned


#  Helper function to make the output dataset
def make_dataset(messy, clean, write, output_filepath):
    df = pd.DataFrame(list(zip(messy, clean)), columns =['messy', 'clean'])

    if write:
        df.to_csv(output_filepath)

    return df


#  Main function to test functionality
def main():
    
    input_filepath = "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Natural Language Processing/Pre-Processing/html_dataset"
    output_filepath = "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Natural Language Processing/Pre-Processing/temp_out.csv"

    messy = get_messy(input_filepath)

    clean = clean_all(messy)

    make_dataset(messy, clean, True, output_filepath)



    
    


if __name__ == "__main__":
    main()
