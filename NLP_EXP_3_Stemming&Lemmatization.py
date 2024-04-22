#!/usr/bin/env python
# coding: utf-8

# In[4]:


import nltk              
nltk.download('omw-1.4') 
from nltk.stem import PorterStemmer, WordNetLemmatizer 
from nltk.corpus import stopwords
nltk.download('punkt')       
nltk.download('wordnet')
nltk.download('stopwords')
text = "The quick brown foxes are jumping over the lazy dogs. They also love running in the fields."
tokens = nltk.word_tokenize(text) 
stemmer = PorterStemmer()      
stemmed_words = [stemmer.stem(word) for word in tokens] 
print("Stemmed Words:")
print(stemmed_words)           
print()
lemmatizer = WordNetLemmatizer() 
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens] 
print("Lemmatized Words:")
print(lemmatized_words)           

