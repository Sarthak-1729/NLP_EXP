#!/usr/bin/env python
# coding: utf-8

# In[13]:


import string
from collections import Counter

def word_frequency_distribution(sentence, words):
    # Define a list of stop words
    stop_words = ["a", "an", "the", "is", "are", "was", "were", "am", "been", "being", "to", "for", "in", "on", "at", "of", "by", "with"]
    
    # Split the sentence into words
    words_in_sentence = sentence.split()
    
    # Remove punctuation from each word and convert to lowercase
    words_in_sentence = [w.strip(string.punctuation).lower() for w in words_in_sentence]
    
    # Remove stop words
    words_in_sentence = [w for w in words_in_sentence if w not in stop_words]
    
    # Create a frequency distribution using Counter
    frequency_distribution = Counter(words_in_sentence)
    
    # Get the frequency of each word in the list
    frequencies = {word: frequency_distribution[word] for word in words}
    
    return frequencies

# Ask the user for input
sentence = input("Enter a sentence: ")
words_to_count = input("Enter words to find their frequency (separated by spaces): ").split()

# Get the word frequency distribution
frequencies = word_frequency_distribution(sentence, words_to_count)

# Print the frequency of each word
print("Frequency of words:")
for word, frequency in frequencies.items():
    print(f"{word}: {frequency}")

