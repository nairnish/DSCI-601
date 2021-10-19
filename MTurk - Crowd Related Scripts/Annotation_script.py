#!/usr/bin/env python
# coding: utf-8

# The idea is to creat bags with different annotation agreement for pt language batch 1 annotation

# In[8]:


'''
import statements here!
'''
import pandas as pd
import csv
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# In[9]:


#Directly reading the result csv downloaded from amazon mechanical turk.
df = pd.read_csv('Result.csv')


# In[10]:


# get the count of "Markers" filled in by the annotators for the instances.
bool_series = pd.notnull(df['Answer.markerVariety'])
y = list(Counter(bool_series).values())
Absent = y[0]
Present = y[1]
Markers = [Absent, Present]
markers=['Absent','Present']


# In[11]:


#Plotting a boxplot for the markers present in the annotation.
plt.bar(range(2), Markers)
plt.xlabel('Markers')
plt.ylabel('Count')
plt.xticks([0,1],markers)
for index, value in enumerate(Markers):
    plt.text(index, value, str(value)) 
plt.show()


# In[12]:


#Grouping the instances by HITId as HITId is same for an instance.
temp_df = df.groupby(df['HITId'])

value_counts = {}
for i in range(len(temp_df.groups)):
    value_counts[list(temp_df.groups.keys())[i]] = df.iloc[list(temp_df.groups.values())[i].values]['Answer.Language Variety.label'].value_counts()


# In[13]:


#Creating csv file from dataframe for poor annotated agreements:
df_poor_ann = pd.DataFrame()
temp_df_poor = pd.DataFrame()
for i in range(len(temp_df.groups)):
    if 1 in list(value_counts.values())[i].values and len(list(value_counts.values())[i].values)==3:
        temp_df_poor = df.iloc[list(temp_df.groups.values())[i].values]
        df_poor_ann = df_poor_ann.append(temp_df_poor, ignore_index=False)
    df_poor_ann  
df_poor_ann.to_csv('ambiguous_annoatation.csv',index=True, header=True)


# In[14]:


#Counting the numbers of different kinds of agreement.
agree_3 = 0
agree_2 = 0
agree_1 = 0
for i in range(len(temp_df.groups)):
    if 2 in list(value_counts.values())[i].values:
        agree_2 += 1
    elif 3 in list(value_counts.values())[i].values:
        agree_3 += 1
    else:
        agree_1 += 1
    


# In[15]:


counts = [agree_3, agree_2, agree_1]
counts_name = ['Excellent', 'Fair', 'Poor']
plt.bar(range(3), counts)
plt.xlabel('inter-annotator agreement')
plt.ylabel('count')
plt.xticks([0,1,2],counts_name)
for index, value in enumerate(counts):
    plt.text(index, value, str(value))
plt.show()

