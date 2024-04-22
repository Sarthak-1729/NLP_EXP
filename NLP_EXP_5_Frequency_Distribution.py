#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
text = """Kratos, his deep voice resonating, spoke to his son Atreus with a stern yet caring tone, "Boy, you must heed my words. We are not gods, nor are we mortals. We walk the path in between, and that demands discipline." Atreus, eager to prove himself, responded with youthful defiance, "But Father, I can handle it! I'm not a child anymore." Kratos, recalling his own tumultuous journey, retorted, "You are not ready. The gods will test you, and the consequences of arrogance are severe." As they ventured through the treacherous realms, the echoes of their dialogues revealed a complex father-son dynamic, blending vulnerability with strength, shaping their epic odyssey in the world of gods and monsters."""
text = re.sub(r'[^\w\s]', '', ' '.join(text.split()))
unique = set([word for word in text.split()])
frequency = {key: text.count(key) for key in unique}
frequency


# In[ ]:




