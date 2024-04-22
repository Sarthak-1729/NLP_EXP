#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.lower() not in stop_words]
    return filtered_text


# In[2]:


from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

def get_root(words):
    lemma_words = [wnl.lemmatize(word, pos = 'v') for word in words]
    lemma_words

    return lemma_words


# In[3]:


import re
import nltk

def preprocess(text):
    text = ' '.join(text.split())
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = remove_stopwords(text)
    root_words = get_root(text)

    return root_words


# In[4]:


text = '''
Once the brutal captain of the Spartan Army, Kratos led his men throughout numerous conquests all across the lands of Greece, eventually coming across a savage Barbarian horde. Confident of his own victory, Kratos led his army into battle, but soon found himself hopelessly outmatched and outclassed. The Barbarians' brutality exceeding his own, and on the verge of death, Kratos struck a deal with the "God of War" - Ares to further his exploits. He would then commit atrocity after atrocity under Ares' name, spreading death throughout the world with his armies and justifying it all by proclaiming his intent to make "the glory of Sparta known throughout the world!". For a time, it seemed, his only tether to humanity was his beloved family, yet even they grew horrified by him, to the point where his wife Lysandra would state outright that he cared nothing for Sparta's glory, but for his own. He would not listen to her, and continued his rampage, blindly following the will of Ares in his pursuit of more bloodshed and infamy — yet this took a tragic turn when the God tricked him into killing his wife and child, all to destroy what little humanity he had left. Branded the "Ghost of Sparta" for this terrible deed, the ashes of his wife and child would remain fused to his skin forever.

Completely undone by the killing of his wife and child, Kratos became a constantly-suicidal and greatly-bereaved wreck of a man beloved by none yet known to all. Devoting himself to the other Gods of Olympus in a desperate attempt to rid himself of his memories, Kratos would hang on to the small glimmer of hope that perhaps he would one day be able to redeem himself. Yet no matter how many enemies he'd slaughter or how many lives he would save, the Gods would continue to put labour upon labour onto Kratos' shoulders, forcing him to endure the pain of his memories for ten long years of servitude. Maddened by his memories and unable to find a moment of peace, Kratos would develop a deep-seated hatred of the Gods, and especially Ares in particular, for toying with his life. Though Kratos would eventually defeat Ares and claim the throne of the "God of War" for his own, his resentment of the other Gods would bring him in conflict against all on Mount Olympus, culminating in a cataclysmic series of battles against them that would decide the fate of Greece itself.

Eventually leaving Greece as well as his bloody past behind, Kratos ends up in Ancient Egypt and makes his way into Midgard. Having come to view his troubled past with great shame, Kratos has taken the initiative to mature and grow past his self-destructive tendencies, choosing to live as a man under the thumb of the Norse Pantheon. He even finds love again with a woman named Faye, eventually fathering a son with her named Atreus. When Faye dies of unknown circumstances, Kratos and Atreus set out on a journey to spread her ashes from the highest peak in all the Nine Realms as it was her final wish. However, he and Atreus come into conflict with various Nordic creatures along their way, and are constantly pursued along their path by a mysterious Stranger — seemingly under orders from the King of the Norse Pantheon himself, Odin.

As Ragnarök unfolds, Kratos finds himself on a difficult journey that places him against the forces of Odin and the friction between him and his son. Later, under Kratos' leadership, all the united forces of the other realms gather throughout Týr's Temple; Kratos blows the Gjallarhorn to begin the siege of Asgard. Initially, the battle does not go well; the other realms are quickly cut off, and Kratos' forces were struggling with Asgard's defenses. After a fight with Thor that ends in his death at the hands of Odin himself, Kratos, along with Atreus have a final battle with Odin and defeats him, resulting in Atreus trapping his soul. Odin is then denied an afterlife by a vengeful Sindri. After defeating Odin and bidding a heartfelt farewell to his son, Kratos discovers a mural depicting him as the new All-Father of Asgard. Finally hopeful about his future, Kratos recruits Freya and Mímir to help him rebuild and restore the Nine Realms.
'''


# In[5]:


pre_text = preprocess(text)
pre_text


# In[6]:


from nltk import pos_tag
from nltk import RegexpParser

tokens_tag = pos_tag(pre_text)
tokens_tag


# In[7]:


entities = [(word, pos) for word, pos in tokens_tag if pos.startswith('N')]


# In[8]:


frequencies = {}
for word, pos in entities:
    if word not in frequencies.keys():
        frequencies[word] = pre_text.count(word)

frequencies


# In[9]:


freqs = []
for key, val in frequencies.items():
    freqs.append((key, val))

freqs.sort(key= lambda x: x[1])


# In[10]:


freqs = freqs[::-1]
freqs[:5]


# In[11]:


set(entities)


# In[12]:


patterns= "NP: {<DT>?<JJ>*<NN>}"
chunker = RegexpParser(patterns)
chunks = chunker.parse(tokens_tag)


# In[13]:


noun_phrases = [subtree.leaves() for subtree in chunks.subtrees() if subtree.label() == 'NP']
noun_phrases


# In[ ]:




