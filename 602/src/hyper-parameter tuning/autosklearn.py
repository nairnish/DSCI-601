# -*- coding: utf-8 -*-
!pip install auto-sklearn
import pandas as pd
from data_load import load_data
from preprocessing import special_character_removal1
from feature_reduce import vectorizer_tfidf_unigram
import autosklearn.classification
from evaluation import sklearn_class_report

train_path = 'DSLCCv4.0_1/DSL-TRAIN.txt'
dev_path = 'DSLCCv4.0_1/DSL-DEV.txt'
test_path = 'DSLCCv4.0_1/DSL-TEST-GOLD.txt'

"""
load data 
@param: path : file path
@return: df : DataFrame with the DSLCC - train,test (gold) or dev
"""
df_train = load_data(train_path)
df_test = load_data(test_path)
df_dev = load_data(dev_path)


X_train = df_train['text']
y_train = df_train['language']


df_train = special_character_removal1(df_train)
df_test = special_character_removal1(df_test)

df_train = df_train[df_train['language'].isin(['pt-BR','pt-PT'])]
df_test = df_test[df_test['language'].isin(['pt-BR','pt-PT'])]

"""
Vectorize using word unigram tf-idf vectorizer 
@param: df_train  : training dataframe
@param: df_test  : testing dataframe
@return: features : sparse matrix vectorized features of train dataset
@return: features_test : sparse matrix vectorized features of test dataset
"""
feature,feature_test = vectorizer_tfidf_unigram(df_train,df_test)

y_train = df_train['language']
y_test = df_test['language']


cls = autosklearn.classification.AutoSklearnClassifier(n_jobs=-1)
cls.fit(feature, y_train)
cls.cv_results_

predictions = cls.predict(feature_test)

print(sklearn_class_report(y_test,predictions))