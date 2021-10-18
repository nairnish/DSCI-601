#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[39]:


class preprocess_data():
    
    __slots__ = "temp_val"
    def __init__(temp_val):
        temp_val = 0
        
    '''
    Stop-word removal for the train
    @param: df: training DataFrame
    @return: df_train: training DataFrame with stopwords removed

    '''
    def stopwords_removal_all(self,df):
        #Stopwords for bosnian, indonesian, spanish, portuguese and french 
        stopwords_list_bosnian = stopwords.words("slovene")
        stopwords_list_indonesian = stopwords.words("indonesian")
        stopwords_list_spanish = stopwords.words("spanish")
        stopwords_list_portuguese = stopwords.words("portuguese")
        stopwords_list_french = stopwords.words("french")
        #combining stopwords into one list
        stopwords_list_combine = stopwords_list_bosnian + stopwords_list_indonesian + stopwords_list_spanish + stopwords_list_portuguese + stopwords_list_french
        newly_filtered_keyword1 = []
        for i in range(len(df)):
            #tokenizing the instances
            tokens = nltk.word_tokenize(df['text'][i])
            #Filtering the stopwords from the combined list
            filtered_text = [t for t in tokens if t not in stopwords_list_combine]
            #Appending new column with each instance
            newly_filtered_keyword1.append(" ".join(filtered_text))
        df['text'] = newly_filtered_keyword1
        return df
    
    '''
    Special Character Removal for the train
    @param: df: training DataFrame
    @return: df_train: training DataFrame with special characters removed
    '''
    def special_character_removal1(self,df):
        #Removal of special characters : cleaning of data.
        spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
              "*", "+", ",", "-", ".", "/", ":", ";", "<",
              "=", ">", "?", "@", "[", "\\", "]", "^", "_",
              "`", "{", "|", "}", "~", "–"]
        for char in spec_chars:
            df['text'] = df['text'].str.replace(char, ' ')
            df['text'] = df['text'].str.replace('https?://\S+|www\.\S+', ' ')
        return df
    
    '''
    Special Character removal + stopword_removal
    @param: df: training DataFrame
    @return: df_train: training DataFrame with special characters and stopwords removed
    '''
    def special_character_stopword_removal(self,df):
        # Removal of special characters
        spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                      "*", "+", ",", "-", ".", "/", ":", ";", "<",
                      "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                      "`", "{", "|", "}", "~", "–"]
        for char in spec_chars:
            df['text'] = df['text'].str.replace(char, ' ')

        # Removal of web-tags
        df['text'] = df['text'].str.replace('https?://\S+|www\.\S+', ' ')

        # Removal of White Spaces
        df['text'] = df['text'].apply(lambda x: x.strip())

        # Removal of Stopwords
        df1 = self.stopwords_removal_all(df)
        return df1
    
    '''
    Stop-word removal for the train
    @param: df: training DataFrame
    @return: df_train: training DataFrame with stopwords removed for Portuguese only 

    '''
    def stopwords_removal_portuguese(self,df):
        #Stopwords for bosnian, indonesian, spanish, portuguese and french 
        #stopwords_list_bosnian = stopwords.words("slovene")
        #stopwords_list_indonesian = stopwords.words("indonesian")
        #stopwords_list_spanish = stopwords.words("spanish")
        stopwords_list_portuguese = stopwords.words("portuguese")
        #stopwords_list_french = stopwords.words("french")
        #combining stopwords into one list
        stopwords_list_combine = stopwords_list_portuguese
        newly_filtered_keyword1 = []
        for i in range(len(df)):
            #tokenizing the instances
            tokens = nltk.word_tokenize(df['text'][i])
            #Filtering the stopwords from the combined list
            filtered_text = [t for t in tokens if t not in stopwords_list_combine]
            #Appending new column with each instance
            newly_filtered_keyword1.append(" ".join(filtered_text))
        df['text'] = newly_filtered_keyword1
        return df
    
    '''
    Stop-word removal for the train
    @param: df: training DataFrame
    @return: df_train: training DataFrame with stopwords removed for Spanish only

    '''
    def stopwords_removal_spanish(self,df):
        #Stopwords for bosnian, indonesian, spanish, portuguese and french 
        #stopwords_list_bosnian = stopwords.words("slovene")
        #stopwords_list_indonesian = stopwords.words("indonesian")
        stopwords_list_spanish = stopwords.words("spanish")
        #stopwords_list_portuguese = stopwords.words("portuguese")
        #stopwords_list_french = stopwords.words("french")
        #combining stopwords into one list
        stopwords_list_combine = stopwords_list_spanish
        newly_filtered_keyword1 = []
        for i in range(len(df)):
            #tokenizing the instances
            tokens = nltk.word_tokenize(df['text'][i])
            #Filtering the stopwords from the combined list
            filtered_text = [t for t in tokens if t not in stopwords_list_combine]
            #Appending new column with each instance
            newly_filtered_keyword1.append(" ".join(filtered_text))
        df['text'] = newly_filtered_keyword1
        return df

