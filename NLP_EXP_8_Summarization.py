#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install sentence-transformers')


# In[2]:


get_ipython().system('pip install gensim')


# In[10]:


from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from nltk.tokenize import sent_tokenize


# In[11]:


def embed_documents(docs) -> list:
  model = SentenceTransformer('all-mpnet-base-v2')
  return model.encode(docs)


# In[12]:


def compute_cosine(A: list, B: list) -> float:
  return np.dot(A,B)/(norm(A)*norm(B))


# In[13]:


def generate_summary(text, similarity_weight = 1, position_weight = 10, num_sentences = 10):
    sentences = sent_tokenize(text)
    clean_sentences = [' '.join(simple_preprocess(strip_tags(sent), deacc=True)) for sent in sentences]
    sentence_vectors = embed_documents(clean_sentences)

    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentence_vectors)):
        position_score = position_weight*(1.0 - (i / len(sentence_vectors)))
        for j in range(len(sentence_vectors)):
            if i != j:
                similarity_matrix[i][j] = similarity_weight*(compute_cosine(sentence_vectors[i], sentence_vectors[j])) + position_score


    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_sentences = [s for _, s in ranked_sentences[:num_sentences]]

    summary = ' '.join(summary_sentences)

    return summary


# In[14]:


import nltk
nltk.download('punkt')


# In[15]:


text = '''
Kratos is a fictional character in the God of War series, a video game franchise developed by Santa Monica Studio and published by Sony Interactive Entertainment. Kratos, a Spartan warrior, serves as the protagonist throughout most of the series. His story is one of tragedy, revenge, and redemption, set in the world of Greek mythology.

The story of Kratos begins with his origins as a mortal warrior in Sparta. He was a skilled and ruthless fighter, eventually rising to become the captain of Sparta's army due to his unmatched prowess in battle. However, Kratos's life took a dark turn when he made a deal with Ares, the Greek god of war. In exchange for power, Kratos pledged his allegiance to Ares, becoming a ruthless and bloodthirsty servant of the god.

Driven by blind rage and fueled by the atrocities he committed under Ares's command, Kratos eventually sought to break free from his servitude. In a moment of clarity, he turned against his master, leading to a brutal conflict that culminated in Kratos killing Ares.

However, this victory did not bring peace to Kratos. Haunted by the memories of his past and plagued by nightmares, he sought to escape the consequences of his actions. In his quest for redemption, Kratos embarked on a journey to the highest peak of Mount Olympus, seeking forgiveness from the gods.

Throughout the series, Kratos faces numerous challenges and battles against various mythical creatures, gods, and titans. His journey takes him across the vast expanse of the Greek world, from the depths of the Underworld to the heights of Olympus itself.

As the series progresses, Kratos's story evolves, delving deeper into themes of family, sacrifice, and the consequences of one's actions. He grapples with his past, his relationships with his family, including his wife, Lysandra, and daughter, Calliope, and the burden of his sins.

In later installments of the series, such as "God of War" (2018), Kratos's story continues in the realm of Norse mythology, where he faces new challenges and confronts his own inner demons while attempting to forge a new path for himself and his son, Atreus.

Overall, the story of Kratos is one of redemption and self-discovery, as he struggles to break free from his past and find a semblance of peace in a world torn apart by gods and monsters.
'''


# In[9]:


generate_summary(text, num_sentences = 3)

