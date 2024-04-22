#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords for multiple languages
nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(text, language):
    stop_words = set(stopwords.words(language))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Example text in English
english_text = "This is an example sentence in English."

# Example text in Spanish
spanish_text = "Este es un ejemplo de frase en español."

# Example text in French
french_text = "Ceci est un exemple de phrase en français."

# Example text in German
german_text = "Dies ist ein Beispiel für einen Satz in Deutsch."

# Example text in Italian
italian_text = "Questo è un esempio di frase in italiano."

# Remove stopwords for each language
english_result = remove_stopwords(english_text, 'english')
spanish_result = remove_stopwords(spanish_text, 'spanish')
french_result = remove_stopwords(french_text, 'french')
german_result = remove_stopwords(german_text, 'german')
italian_result = remove_stopwords(italian_text, 'italian')

# Print the results
print("English:", english_result)
print("Spanish:", spanish_result)
print("French:", french_result)
print("German:", german_result)
print("Italian:", italian_result)


# In[ ]:




