#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fasttext as ft
import pandas as pd
import time

start = time.time()

##model training
model = ft.train_supervised(input="fasttext_es_train.txt", wordNgrams=2 ) 

##model prediction on test data
model.test("fasttext_es_test.txt")


# In[2]:


model_runtime = (time.time() - start)
print(model_runtime)

