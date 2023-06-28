"""
Author: Rohan Singh
Code for MLM
"""

# Imports
from transformers import pipeline

# Main function
def main():
    unmasker = pipeline('fill-mask', model='bert-base-uncased')

    prompt = "rohan and rachel should [MASK]."

    print("\nThe Prompt is:\n" + prompt + "\n")

    responses = unmasker(prompt)

    print("\nThe responses are:")
    for response in responses:
        print(response) 
    
    """
    prompt = "rohan works as a [MASK]."

    print("\nThe Prompt is:\n" + prompt + "\n")

    responses = unmasker(prompt)

    print("\nThe responses are:")
    for response in responses:
        print(response)
    """

    

if __name__ == "__main__":
    main()