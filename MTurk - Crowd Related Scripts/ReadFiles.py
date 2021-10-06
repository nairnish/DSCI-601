#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd


# In[19]:


##Load train data
train_data = pd.read_csv('/Users/neha/Documents/Semester_2/601TrailRuns/v3.0_train/task1-train.txt', delimiter = "\t", header = None)


# In[20]:


len(train_data)


# In[21]:


##Renaming column name.
train_data = train_data.rename(columns = {0 : "text"})
train_data = train_data.rename(columns = {1 : "lang_variant"})
##Training the model just for portuguese variant.
#portugese_var =['pt-BR', 'pt-PT']
portugese_var =['pt-BR']
#portugese_var =['pt-PT']
train_pt_BR = train_data[train_data['lang_variant'].isin(portugese_var)]


# In[22]:


len(train_pt_BR)


# In[23]:


numpy_array = train_pt_BR.to_numpy()
np.savetxt("train_pt_BR.txt", numpy_array, fmt = "%s")


# In[24]:


train_pt_BR.to_csv("train_pt_BR.csv", index=True)


# In[25]:


##Renaming column name.
train_data = train_data.rename(columns = {0 : "text"})
train_data = train_data.rename(columns = {1 : "lang_variant"})
##Training the model just for portuguese variant.
#portugese_var =['pt-BR', 'pt-PT']
#portugese_var =['pt-BR']
portugese_var =['pt-PT']
train_pt_PT = train_data[train_data['lang_variant'].isin(portugese_var)]


# In[26]:


len(train_pt_PT)


# In[27]:


numpy_array = train_pt_PT.to_numpy()
np.savetxt("train_pt_PT.txt", numpy_array, fmt = "%s")


# In[28]:


train_pt_PT.to_csv("train_pt_PT.csv", index=True)


# In[29]:


#Creation of Bag1
bag1 = pd.concat([train_pt_BR[0:1250],train_pt_PT[0:1250]])


# In[30]:


bag1['id'] = (bag1.index.values)


# In[31]:


bag1 = bag1.reindex(columns=['id','text','lang_variant'])


# In[32]:


bag1_shuffle = bag1.sample(frac=1)
len(bag1_shuffle)


# In[33]:


numpy_array = bag1.to_numpy()
np.savetxt("bag1.txt", numpy_array, fmt = "%s")
bag1.to_csv("bag1.csv", index=True)
numpy_array = bag1_shuffle.to_numpy()
np.savetxt("bag1_shuffle.txt", numpy_array, fmt = "%s")
bag1_shuffle.to_csv("bag1_shuffle.csv", index=True)


# In[34]:


bag1.head


# In[35]:


bag1_shuffle.head


# In[36]:


#Creation of Bag2
bag2 = pd.concat([train_pt_BR[1250:2500],train_pt_PT[1250:2500]])


# In[39]:


bag2_shuffle = bag2.sample(frac=1)
len(bag2_shuffle)


# In[40]:


bag2_no_label = bag2.drop('lang_variant', axis = 1)


# In[41]:


bag2_shuffle_no_label = bag2_no_label.sample(frac=1)
len(bag2_shuffle_no_label)


# In[42]:


len(bag2)


# In[43]:


numpy_array = bag2_no_label.to_numpy()
np.savetxt("bag2_no_label.txt", numpy_array, fmt = "%s")
bag2_no_label.to_csv("bag2_no_label.csv", index=True)
numpy_array = bag2_shuffle_no_label.to_numpy()
np.savetxt("bag2_shuffle_no_label.txt", numpy_array, fmt = "%s")
bag2_shuffle_no_label.to_csv("bag2_shuffle_no_label.csv", index=True)


# In[44]:


numpy_array = bag2.to_numpy()
np.savetxt("bag2.txt", numpy_array, fmt = "%s")
bag2.to_csv("bag2.csv", index=True)
numpy_array = bag2_shuffle.to_numpy()
np.savetxt("bag2_shuffle.txt", numpy_array, fmt = "%s")
bag2_shuffle.to_csv("bag2_shuffle.csv", index=True)


# In[45]:


bag2.head


# In[46]:


bag1_no_label = bag1.drop('lang_variant', axis = 1)


# In[47]:


bag1_shuffle_no_label = bag1_no_label.sample(frac=1)
len(bag1_shuffle_no_label)


# In[48]:


numpy_array = bag1_no_label.to_numpy()
np.savetxt("bag1_no_label.txt", numpy_array, fmt = "%s")
bag1_no_label.to_csv("bag1_no_label.csv", index=True)
numpy_array = bag1_shuffle_no_label.to_numpy()
np.savetxt("bag1_shuffle_no_label.txt", numpy_array, fmt = "%s")
bag1_shuffle_no_label.to_csv("bag1_shuffle_no_label.csv", index=True)


# In[ ]:




