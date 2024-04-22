#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install nltk')
import nltk


# In[5]:


from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[6]:


nltk.download('punkt')
nltk.download('stopwords')


# In[7]:


def preprocess(text):
    tokens = word_tokenize(text.lower()) 
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens


# In[8]:


def vectorize(tokens):
    vectorizer = CountVectorizer()
    token_matrix = vectorizer.fit_transform([" ".join(tokens), text])
    
    return token_matrix


# In[9]:


def summarize(text, top_n=2):

    tokens = preprocess(text)

    token_matrix = vectorize(tokens)

    similarity = cosine_similarity(token_matrix)[0] # Calculate similarity between the tokenized text and each sentence
    print("Vector Similarity Scores:")
    for i, score in enumerate(similarity):
        print(f"Sentence {i+1}: {score}")

    top_indices = similarity.argsort()[-top_n:][::-1] # Get indices of most similar sentences
    sentences = sent_tokenize(text)
    summary = [sentences[i] for i in top_indices] # Extract most similar sentences

    return ' '.join(summary)


# In[10]:


text = """
Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines, particularly computer systems, as opposed to the natural intelligence of living beings. It is a field of research in computer science that develops and studies methods and software which enable machines to perceive their environment and uses learning and intelligence to take actions that maximize their chances of achieving defined goals."""


# In[11]:


summary = summarize(text)
print("\nSummary:")
print(summary)


# In[ ]:




