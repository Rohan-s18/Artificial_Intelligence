"""
Author: Rohan Singh
Python Module to make an html dataset
Date: 28 May 2023
"""



#  Imports
import os
import random
from string import ascii_lowercase
import nltk
from nltk.corpus import brown

nltk.download('punkt')
nltk.download('brown')
sentences = brown.sents()

#  Helper function to make the dataset
def make_dataset(output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i in range(1, 1001):
        filename = f"file_{i}.html"
        file_path = os.path.join(output_directory, filename)

        with open(file_path, "w") as file:
            file.write("<html><body>")
        
            # Add random heading tag
            headings = ["h1", "h2", "h3"]
            heading = random.choice(headings)
            file.write(f"<{heading}>File {i}</{heading}>")
        
            # Add random number of paragraphs
            num_paragraphs = random.randint(2, 5)
            for _ in range(num_paragraphs):
                # Add random paragraph content
                paragraph_content = ' '.join(random.choice(sentences))
                file.write(f"<p>{paragraph_content}</p>")
        
            file.write("</body></html>")

    print("HTML dataset created successfully!")


#  Main function to test functionality
def main():
    output_directory = "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Natural Language Processing/Pre-Processing/html_dataset"

    make_dataset(output_directory)


if __name__ == "__main__":
    main()
