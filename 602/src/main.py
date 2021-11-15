# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Dof1nmyNGqNQD97zqKsXhqlOojpHcn0E
"""

#Importing the necessary libraries
from data_load import load_data
from preprocessing import preprocess as preprocess_data
from feature_reduce import feature_reduction 
from models import classical_models
from evaluation import evaluation

if name=="__main__":
   train_path = 'data/DSL-TRAIN.txt'
   test_path = 'data/DSL-TEST-GOLD.txt'
   dev_path = 'data/DSL-DEV.txt'
   print("Loading the data files: ")
   df_train = load_data(train_path)
   df_test = load_data(test_path)
   df_val = load_data(val_path)
   print("Preprocessing...")
   int_p = input("What kind of preprocessing do you want? 1 for stopword removal, 2 for special character removal, 3 for stopword removal (portuguese), 4 for stopword removal (spanish), 5 for stopword and special character removal.")
   if (int_p == 1):
     df_train = preprocess_data.stopword_removal_all(df_train)
     df_test = preprocess_data.stopword_removal_all(df_test)
   elif (int_p == 2):
     df_train,df_test = preprocess_data.special_character_removal1(df_train), preprocess_data.special_character_removal1(df_test)
   elif (int_p == 3):
     df_train,df_test = preprocess_data.stopword_removal_portuguese(df_train),preprocess_data.stopword_removal_portuguese(df_test)
   elif (int_p == 4):
     df_train,df_test = preprocess_data.stopword_removal_spanish(df_train),preprocess_data.stopword_removal_spanish(df_test)
   else:
     df_train,df_test = preprocess_data.special_character_stopword_removal(df_train),preprocess_data.special_character_stopword_removal(df_test)
   print("Feature reduction...")
   int_fr = input("What kind of features do you want? 1 for word unigrams + bigrams , 2 for word unigrams , 3 for character 4 grams")
   if (int_fr == 1):
      feature_train,feature_test = feature_reduction.vectorizer_tfidf_unigram_bigram(df_train,df_test)
   elif (int_fr == 2):
      feature_train,feature_test = feature_reduction.vectorizer_tfidf_unigram(df_train,df_test)
   elif (int_fr == 3):
      feature_train,feature_test = feature_reduction.vectorizer_tfidf_4gram(df_train,df_test)
   print("Modelling...")
   print("SVC model...")
   predictions = model_SVC(feature_train,feature_test)
   print("Evaluation...")
   print(evaluation.sklearn_class_report(df_test['language'],predictions))

