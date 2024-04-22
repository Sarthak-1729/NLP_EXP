#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install nltk


# In[5]:


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize, TweetTokenizer

# Sample text for demonstration
sample_text = "NLP LAB 1 EXP 1 apply different tokenization techniques using nltk. Will learn about natural language tool-kit."

# Word Tokenization
words = word_tokenize(sample_text)
print("Word Tokenization:")
print(words)

# Sentence Tokenization
sentences = sent_tokenize(sample_text)
print("\nSentence Tokenization:")
print(sentences)

# Regular Expression Tokenization
# Tokenize based on words containing alphabets
regexp_tokens = regexp_tokenize(sample_text, pattern=r'\b\w+\b')
print("\nRegExp Tokenization (Alphabets Only):")
print(regexp_tokens)

# Tokenize based on words containing alphabets and digits
regexp_tokens_with_digits = regexp_tokenize(sample_text, pattern=r'\b\w+\b|\d+')
print("\nRegExp Tokenization (Alphabets and Digits):")
print(regexp_tokens_with_digits)

# Custom Tokenization (e.g., splitting by hyphens)
custom_tokens = sample_text.split('-')
print("\nCustom Tokenization (Splitting by Hyphens):")
print(custom_tokens)

# Tweet Tokenization (Handles mentions, hashtags, and URLs)
tweet_tokenizer = TweetTokenizer()
tweet_tokens = tweet_tokenizer.tokenize("This is a #tweet with @mentions and a URL: https://example.com")
print("\nTweet Tokenization:")
print(tweet_tokens)


# In[ ]:




