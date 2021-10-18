#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[11]:


class feature_reduction():
    
    __slots__ = "temp_val"
    def __init__(temp_val):
        temp_val = 0
    
    '''
    Tfidf Vectorizer 

    @param: df_train : Training dataFrame
    @param: df_test_gold: Testing dataFrame withn gold standards

    @return: feature: training feature set
    @return: feature_test: testing feature set

    '''
    def vectorizer_tfidf(self,df_train,df_test_gold):
        vectorizer = TfidfVectorizer() 
        vectorizer.fit(df_train['text'])
        feature = vectorizer.transform(df_train['text'])
        feature_test = vectorizer.transform(df_test_gold['text'])
        return feature,feature_test
        
    '''
    Column Transformer (still having doubts about this)
    @param: df
    @return: feature_set 
    '''
    def column_transformer(df):
        
        return feature_set

