#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[2]:


text = """
There are no stars tonight
But those of memory.
Yet how much room for memory there is
In the loose girdle of soft rain.
There is even room enough
For the letters of my mother’s mother,
Elizabeth,
That have been pressed so long
Into a corner of the roof
That they are brown and soft,
And liable to melt as snow.
Over the greatness of such space
Steps must be gentle.
It is all hung by an invisible white hair.
It trembles as birch limbs webbing the air.
"""


# In[3]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1


# In[4]:


input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


# In[5]:


max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))


# In[6]:


predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
label = tf.keras.utils.to_categorical(label, num_classes=total_words)


# In[7]:


model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[8]:


model.fit(predictors, label, epochs=500, verbose=1)


# In[9]:


def generate_text(seed_text, next_words, model, max_sequence_len, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted_probs = model.predict(token_list, verbose=0).flatten()

        predicted_probs = np.log(predicted_probs) / temperature
        predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))

        predicted = np.random.choice(range(total_words), size=1, p=predicted_probs)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


# In[10]:


seed_text = "From fairest creatures we desire"
generated_text = generate_text(seed_text, 10, model, max_sequence_len, temperature=0.5)
print(generated_text)


# In[ ]:




