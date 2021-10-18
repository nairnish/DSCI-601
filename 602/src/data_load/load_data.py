#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


class load_data():
    '''
    Pre-process data 

    @param: path : file path
    @return: df : DataFrame with the DSLCC - train,test (gold) or dev

    '''
    def load(path):
        data = []
        #Reading contents from file 
        file_contents = open(path,"r",encoding="utf-8")
        #Appending read data to the file
        data.append(file_contents.read())
        text = []
        language = []
        #splitting each instance by \n
        temp = data[0].split("\n")
        for i in range(len(temp)):
            #putting each instance's text into one column
            text.append(temp[i].split("\t")[0])
            #putting language variety into second column
            language.append(temp[i].split("\t")[1])
        #making the DataFrame    
        df = pd.DataFrame(data={'text':text,'language':language})
        return df

