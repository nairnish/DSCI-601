#!/usr/bin/env python
# coding: utf-8

# # Classification of All labels 

# Importing the necessary libraries




import pandas as pd
import os 
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier


# Define paths for train,dev and test-gold datasets




train_path = 'C:/Users/Yash/Downloads/DSLCCv4.0_1/DSL-TRAIN.txt'
dev_path = 'C:/Users/Yash/Downloads/DSLCCv4.0_1/DSL-DEV.txt'
test_path = 'C:/Users/Yash/Downloads/DSLCCv4.0_1/DSL-TEST-GOLD.txt'





"""
Pre-process data 

@param: path : file path
@return: df : DataFrame with the DSLCC - train,test (gold) or dev

"""
def preprocess(path):
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


"""
Stop-word removal for the train

@param: df: training DataFrame

@return: df_train: training DataFrame with stopwords removed

"""
def stopwords_removal(df):
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
    df['newly_filtered_keyword'] = newly_filtered_keyword1
    return df




"""
Tfidf Vectorizer 

@param: df_train : Training dataFrame
@param: df_test_gold: Testing dataFrame withn gold standards

@return: feature: training feature set
@return: feature_test: testing feature set

"""
def vectorizer_tfidf(df_train,df_test_gold):
    vectorizer = TfidfVectorizer() 
    vectorizer.fit(df_train['newly_filtered_keyword'])
    feature = vectorizer.transform(df_train['newly_filtered_keyword'])
    feature_test = vectorizer.transform(df_test_gold['text'])
    return feature,feature_test




"""
Classification Report and Time to Run Support Vector Machine

@param: df_train: Training dataFrame 
@param: df_test_gold: Testing dataFrame with gold standard

@return: void 

"""
def fit_predict_SVC(df_train,df_test):
    clf_SVC = SVC()
    start_time = time.time()
    df_train = stopwords_removal(df_train)
    df_test_gold = stopwords_removal(df_test)
    feature,feature_test = vectorizer_tfidf(df_train,df_test_gold)
    clf_SVC.fit(feature,df_train['language'])
    predictions = clf_SVC.predict(feature_test)
    print(classification_report(df_test_gold['language'],predictions))
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken to run Support Vector Machine is %f seconds" %time_taken)
    return



"""
Classification Report and Time to Run K Nearest Neighbors

@param: df_train: Training dataFrame 
@param: df_test_gold: Testing dataFrame with gold standard

@return: void 

"""
def fit_predict_KNN(df_train,df_test):
    clf_KNN = KNeighborsClassifier(n_neighbors=10)
    start_time = time.time()
    df_train = stopwords_removal(df_train)
    df_test_gold = stopwords_removal(df_test)
    feature,feature_test = vectorizer_tfidf(df_train,df_test_gold)
    clf_KNN.fit(feature,df_train['language'])
    predictions = clf_KNN.predict(feature_test)
    print(classification_report(df_test_gold['language'],predictions))
    end_time = time.time()
    time_taken = end_time-start_time
    print("Time taken to run K Neighbors Classifier is %f seconds" %time_taken)
    return




"""
Classification Report and Time to Run Decision Tree Classifier

@param: df_train: Training dataFrame 
@param: df_test_gold: Testing dataFrame with gold standard

@return: void 

"""
def fit_predict_DTC(df_train,df_test):
    clf_dtc = DecisionTreeClassifier()
    start_time = time.time()
    df_train = stopwords_removal(df_train)
    df_test_gold = stopwords_removal(df_test)
    feature,feature_test = vectorizer_tfidf(df_train,df_test_gold)
    clf_dtc.fit(feature,df_train['language'])
    predictions = clf_dtc.predict(feature_test)
    print(classification_report(df_test_gold['language'],predictions))
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken to run Decision Tree Classifier is %f seconds" %time_taken)
    return




"""
Classification Report and Time to Run Random Forest Classifier

@param: df_train: Training dataFrame 
@param: df_test_gold: Testing dataFrame with gold standard

@return: void 

"""
def fit_predict_RFC(df_train,df_test):
    clf_RFC = RandomForestClassifier()
    start_time = time.time()
    df_train = stopwords_removal(df_train)
    df_test_gold = stopwords_removal(df_test)
    feature,feature_test = vectorizer_tfidf(df_train,df_test_gold)
    clf_RFC.fit(feature,df_train['language'])
    predictions = clf_RFC.predict(feature_test)
    print(classification_report(df_test_gold['language'],predictions))
    end_time = time.time()
    time_taken = end_time-start_time
    print("Time taken to run Random Forest Classifier is %f seconds" %time_taken)
    return




"""
Classification Report and Time to Run AdaBoost Classifier

@param: df_train: Training dataFrame 
@param: df_test_gold: Testing dataFrame with gold standard

@return: void 

"""
def fit_predict_Ada(df_train,df_test):
    clf_ada = AdaBoostClassifier()
    start_time = time.time()
    df_train = stopwords_removal(df_train)
    df_test_gold = stopwords_removal(df_test)
    feature,feature_test = vectorizer_tfidf(df_train,df_test_gold)
    clf_ada.fit(feature,df_train['language'])
    predictions = clf_ada.predict(feature_test)
    print(classification_report(df_test_gold['language'],predictions))
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken to run Ada Boost Classifier is %f seconds" %time_taken)
    return




"""
Classification Report and Time to Run ExtraTreesClassifier

@param: df_train: Training dataFrame 
@param: df_test_gold: Testing dataFrame with gold standard

@return: void 

"""
def fit_predict_Extra(df_train,df_test):
    clf_extra = ExtraTreesClassifier()
    start_time = time.time()
    df_train = stopwords_removal(df_train)
    df_test_gold = stopwords_removal(df_test)
    feature,feature_test = vectorizer_tfidf(df_train,df_test_gold)
    clf_extra.fit(feature,df_train['language'])
    predictions = clf_extra.predict(feature_test)
    print(classification_report(df_test_gold['language'],predictions))
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken to run Extra Trees Classifier is %f seconds" %time_taken)
    return



"""
Classification Report and Time to Run SGDClassifier

@param: df_train: Training dataFrame 
@param: df_test_gold: Testing dataFrame with gold standard

@return: void 

"""
def fit_predict_SGDC(df_train,df_test):
    clf_sgdc = SGDClassifier()
    start_time = time.time()
    df_train = stopwords_removal(df_train)
    df_test_gold = stopwords_removal(df_test)
    feature,feature_test = vectorizer_tfidf(df_train,df_test_gold)
    clf_sgdc.fit(feature,df_train['language'])
    predictions = clf_sgdc.predict(feature_test)
    print(classification_report(df_test_gold['language'],predictions))
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken to run Stochastic Gradient Descent Classifier is %f seconds" %time_taken)
    return


#testing the functions 
df_train = preprocess(train_path)
df_test = preprocess(test_path)
fit_predict_SVC(df_train,df_test)
fit_predict_KNN(df_train,df_test)
fit_predict_DTC(df_train,df_test)
fit_predict_RFC(df_train,df_test)
fit_predict_Ada(df_train,df_test)
fit_predict_Extra(df_train,df_test)
fit_predict_SGDC(df_train,df_test)

