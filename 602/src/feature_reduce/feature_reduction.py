#!/usr/bin/env python
# coding: utf-8
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

class feature_reduction():
    
    __slots__ = "temp_val"
    def __init__(temp_val):
        temp_val = 0
    
    '''
    Tfidf Vectorizer word unigram

    @param: df_train : Training dataFrame
    @param: df_test_gold: Testing dataFrame withn gold standards

    @return: feature: training feature set
    @return: feature_test: testing feature set

    '''
    def vectorizer_tfidf_unigram(self,df_train,df_test_gold):
        vectorizer = TfidfVectorizer()
        vectorizer.fit(df_train['text'])
        feature = vectorizer.transform(df_train['text'])
        feature_test = vectorizer.transform(df_test_gold['text'])
        return feature,feature_test

    '''
    Tfidf Vectorizer word unigram-bigram
    
    @param: df_train : Training dataFrame
    @param: df_test_gold: Testing dataFrame withn gold standards
    
    @return: feature: training feature set
    @return: feature_test: testing feature set
    '''
    def vectorizer_tfidf_unigram_bigram(self, df_train, df_test_gold):
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        vectorizer.fit(df_train['text'])
        feature = vectorizer.transform(df_train['text'])
        feature_test = vectorizer.transform(df_test_gold['text'])
        return feature, feature_test

    '''
     Tfidf Vectorizer with  character 4-gram

    @param: df_train : Training dataFrame
    @param: df_test_gold: Testing dataFrame withn gold standards

    @return: feature: training feature set
    @return: feature_test: testing feature set
    '''

    def vectorizer_tfidf_4gram(self,df_train,df_test_gold):
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(4, 4))
        vectorizer.fit(df_train['text'])
        feature = vectorizer.transform(df_train['text'])
        feature_test = vectorizer.transform(df_test_gold['text'])
        return feature,feature_test



    '''
    Column Transformer
    @param: df
    @return: feature_set 
    '''
    def column_transformer(df):
        # Only keeping required text features
        required_text_features = ['new_text']
        # Vectorizing text data using TF-IDF vectorizer
        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1,2)))])
        preprocessor = ColumnTransformer(
            transformers=[
                *[(feature_name, text_transformer, feature_name)
                  for feature_name in required_text_features]
            ]
        )

        return preprocessor

