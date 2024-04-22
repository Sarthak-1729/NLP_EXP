#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install nltk


# In[3]:


import nltk
nltk.download('punkt')  # Download the necessary NLTK data

from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser

# Sample sentence with more complex structure
sentence = "The quick brown fox jumps over the lazy dog, and the cat watches from the windowsill."

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Perform PoS tagging
pos_tags = pos_tag(tokens)

# Define a chunking grammar
grammar = r"""
    NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
    VP: {<VB.*><NP|PP|CLAUSE>+$}  # Chunk verbs and their arguments
    CLAUSE: {<NP><VP>}            # Chunk NP, VP pairs
"""

# Create a chunk parser with the defined grammar
chunk_parser = RegexpParser(grammar)

# Parse the PoS tagged sentence
chunks = chunk_parser.parse(pos_tags)

# Output the original sentence, PoS tags, and chunked sentence
print("Original Sentence:", sentence)
print("\nPoS Tags:", pos_tags)
print("\nChunked Sentence:")
print(chunks)


# In[ ]:




