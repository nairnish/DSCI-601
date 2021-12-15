
import pandas as pd
from tpot import TPOTClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from data_load import load_data
From feature_reduce import feature_reduction

train_path = 'data/DSL-TRAIN.txt'
test_path = 'data/DSL-TEST-GOLD.txt'

"""
load data 
@param: path : file path
@return: df : DataFrame with the DSLCC - train,test (gold) or dev
"""
data = load_data()
df_train = data.load(train_path)
df_test = data.load(test_path)


##Training the model just for portuguese variant.
df_train = df_train[df_train["language"].isin(["pt-BR","pt-PT"])]
df_test =  df_test[df_test["language"].isin(["pt-BR","pt-PT"])]



"""
Vectorize using word unigram tf-idf vectorizer 
@param: df_train  : training dataframe
@param: df_test  : testing dataframe
@return: features : sparse matrix vectorized features of train dataset
@return: features_test : sparse matrix vectorized features of test dataset
"""
vectorizer = feature_reduction()


feature_train, feature_test = vectorizer.vectorizer_tfidf(df_train,df_test)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, n_jobs=-1, config_dict = 'TPOT sparse')
tpot.fit(feature_train, df_train['language'])
print(tpot.score(feature_test, df_test['language']))

