#!/usr/bin/env python
# coding: utf-8

# In[4]:


import string
from collections import Counter

def word_frequency_distribution(review, words):
    # Define a list of stop words
    stop_words = ["a", "an", "the", "is", "are", "was", "were", "am", "been", "being", "to", "for", "in", "on", "at", "of", "by", "with"]
    
    # Split the review into words
    words_in_review = review.split()
    
    # Remove punctuation from each word and convert to lowercase
    words_in_review = [w.strip(string.punctuation).lower() for w in words_in_review]
    
    # Remove stop words
    words_in_review = [w for w in words_in_review if w not in stop_words]
    
    # Create a frequency distribution using Counter
    frequency_distribution = Counter(words_in_review)
    
    # Get the frequency of each word in the list
    frequencies = {word: frequency_distribution[word] for word in words}
    
    return frequencies

# Get the movie review from the user
review = input("Enter your movie review: ")

# Get words to find their frequency in the review
words_to_count = input("Enter words to find their frequency in the review (separated by spaces): ").split()

# Get the word frequency distribution
frequencies = word_frequency_distribution(review, words_to_count)

# Print the frequency of each word
print("Frequency of words in the review:")
for word, frequency in frequencies.items():
    print(f"{word}: {frequency}")

