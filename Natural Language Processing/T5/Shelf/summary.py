"""
This Script contains code for abstractive summarization using T5 off the shelf
Author: Rohan Singh
Date: Aug 15, 2023
"""

# Imports
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

base_model = T5ForConditionalGeneration.from_pretrained('t5-base')
base_tokenizer = T5Tokenizer.from_pretrained('t5-base')

text_to_summarize ="""I am a Computer Science, Physics & Math student at Case Western Reserve University in Cleveland, Ohio. I am currently working as a Machine Learning Intern at Lockheed Martin & I plan on attending Graduate School after this. I am interested in Quantum Computing, Artificial Intelligence, Machine Learning, Operating System and Low-level programming. If you're interested in working on a project with me, please feel free to reach out. I also love watching baseball in my free time (San Francisco Giants Fan).
I am undertaking a research project on a Drug-Drug interaction and Drug-Target interactions using Coupled Tensor-Tensor Completion with the Department of Computer Science and Data Science at Case Western Reserve University. My other research work deals with using Transformer based models for Sequence and Token Classification and using that for Predictive Analysis. Some of my side projects include designing and programming embedded systems.
"""

preprocess_text = text_to_summarize.strip().replace("\n","")

print ("original text preprocessed:\n", preprocess_text)

# known prompt for summarization with T5
t5_prepared_text = "summarize: " + preprocess_text

input_ids = base_tokenizer.encode(t5_prepared_text, return_tensors="pt")

# summmarize 
summary_ids = base_model.generate(
    input_ids,
    num_beams=4,
    no_repeat_ngram_size=3,
    min_length=30,
    max_length=50,
    early_stopping=True
)

output = base_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print (f"Summarized text: \n{output}")