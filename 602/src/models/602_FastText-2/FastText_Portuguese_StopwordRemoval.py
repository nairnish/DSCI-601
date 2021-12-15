#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# # Classification of Portuguese Language Variants 
# Importing the necessary libraries
import pandas as pd
import os 
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
import fasttext 
import time
from .data_load import load_data as pre-process
from .preprocessing import stopwords_removal_all as stopwords_removal

# Define paths for train,dev and test-gold datasets
train_path = '../../../data/DSL-TRAIN.txt'
dev_path = '../../../data/DSL-DEV.txt'
test_path = '../../../data/DSL-TEST-GOLD.txt'




"""
Convert Training DataFrame to desired Training format (text file) for input into FastText
@param: df_stopwords_removed: training DataFrame with stopwords removed
@param: filename: Name of the file as filename.txt which is string
@return: void
"""
def convert_data(df,filename):
    file = open(filename,"w+",encoding='utf-8')
    for i in df.index:
        lines= '__label__'+str(df['language'][i])+ " " + df['text'][i]
        file.write(lines+'\n')
    return 

"""
Train and Predict using FastText
@param: filename: input file for FastText model
@param: filename_test: file for testing the trained model
@return: void
"""
def fastText_train_predict(filename,filename_test):
    model = fasttext.train_supervised(filename)
    print("The model trained on the following labels: ")
    print(model.labels)
    print_results(*model.test(filename_test))
    return  

"""
Print results in nice format
@params: N : number of test instances
@params: p : Precision
@params: r : Recall
@return: void 
"""
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    return 


# In[4]:


#Pre-processing the train and test
df_train = preprocess(train_path)
df_train = stopwords_removal(df_train)
df_train = convert_data(df_train,"FastText_Portuguese_StopwordsRemoval_train.txt")
df_test = preprocess(test_path)
df_test = stopwords_removal(df_test)
df_test = convert_data(df_test,"FastText_Portuguese_StopwordsRemoval_test.txt")


# In[5]:


#Time taken - FastText implementation
start = time.time()
fastText_train_predict("FastText_Portuguese_StopwordsRemoval_train.txt","FastText_Portuguese_StopwordsRemoval_test.txt")
end = time.time()
print("Time in s: %f s" %(end-start))


# In[ ]:




