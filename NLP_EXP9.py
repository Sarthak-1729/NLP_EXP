#!/usr/bin/env python
# coding: utf-8

# In[10]:


pip install tensorflow


# In[11]:


poem = '''
Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could
To where it bent in the undergrowth;

Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same,

And both that morning equally lay
In leaves no step had trodden black.
Oh, I kept the first for another day!
Yet knowing how way leads on to way,
I doubted if I should ever come back.

I shall be telling this with a sigh
Somewhere ages and ages hence:
Two roads diverged in a wood, and I—
I took the one less traveled by,
And that has made all the difference.
'''


# In[12]:


lines = poem.split('\n')


# In[13]:


lines = lines[1:]
lines


# In[14]:


import string

def preprocess(text):
    text = ''.join(char for char in text if char not in string.punctuation).lower()

    return text

dataset = [preprocess(line) for line in lines]
dataset


# In[15]:


from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(oov_token='<NULL>')
tokenizer.fit_on_texts(dataset)


# In[16]:


tokenizer.word_index


# In[17]:


tokenizer.word_counts


# In[18]:


tokenized_dataset = [tokenizer.texts_to_sequences([line]) for line in dataset]


# In[19]:


tokenized_dataset[0]


# In[20]:


inp_sequences = []

for sequence in tokenized_dataset:
    for i, token in enumerate(sequence[0]):
        ngram_sequence = sequence[0][0:i+2]
        inp_sequences.append(ngram_sequence)
inp_sequences


# In[21]:


import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def generate_padded_dataset(inp_sequences):
    max_len = max([len(x) for x in inp_sequences])
    padded_sequences = np.array(pad_sequences(inp_sequences,max_len, padding = 'pre'))

    predictors, labels = padded_sequences[:,:-1], padded_sequences[:,-1]
    labels = to_categorical(labels, num_classes=len(tokenizer.word_index) + 1)

    return padded_sequences, predictors, labels

padded_sequences, predictors, labels = generate_padded_dataset(inp_sequences)


# In[22]:


predictors[0]


# In[23]:


from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential

max_len = max([len(x) for x in inp_sequences])
total_words = len(tokenizer.word_index) + 1

def create_model(max_len, total_words):
    input_length = max_len - 1
    model = Sequential()

    model.add(Embedding(total_words,512, input_length=input_length))
    model.add(LSTM(256))
    model.add(Dropout(0.1))

    model.add(Dense(total_words,activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer='adam')

    return model

model = create_model(max_len, total_words)
model.summary()


# In[24]:


model.fit(predictors, labels, epochs=100, verbose=5)


# In[25]:


def generate_text(seed_text, num_words, model, max_len):
    for _ in range(num_words):
        tokens = tokenizer.texts_to_sequences([seed_text])[0]
        padded_tokens = pad_sequences([tokens], maxlen=max_len -1, padding='pre')
        predicted = model.predict(padded_tokens, verbose = 0)

        position = np.argmax(predicted[0])

        output = ''
        for word, index in tokenizer.word_index.items():
            if index == position:
                output = word
                break
        
        seed_text += " " + output
    
    return seed_text.title()


# In[28]:


generate_text('TEXT - ', 8, model, max_len)


# In[ ]:




