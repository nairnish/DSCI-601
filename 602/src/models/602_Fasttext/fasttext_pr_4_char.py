#!/usr/bin/env python
# coding: utf-8

# In[14]:


import fasttext as ft
import pandas as pd
import time

start = time.time()

##model training
model = ft.train_supervised(input="fasttext_pr_train.txt", minn=4,maxn=4 ) 

##model prediction on test data
model.test("fasttext_pr_test.txt")


# In[15]:


model_runtime = (time.time() - start)
print(model_runtime)


# In[ ]:




