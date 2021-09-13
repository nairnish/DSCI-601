#!/usr/bin/env python
# coding: utf-8

# In[27]:


#!/usr/bin/env python
# coding: utf-8

# # Classification of All labels 

# Importing the necessary libraries
import pandas as pd
import os 
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
import fasttext 
import time


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


# In[28]:


#Pre-processing the train and test
df_train = preprocess(train_path)
df_train = stopwords_removal(df_train)
df_train = convert_data(df_train,"FastText_All_StopwordsRemoval_train.txt")
df_test = preprocess(test_path)
df_test = stopwords_removal(df_test)
df_test = convert_data(df_test,"FastText_All_StopwordsRemoval_test.txt")


# In[30]:


#Time taken - FastText implementation
start = time.time()
fastText_train_predict("FastText_All_StopwordsRemoval_train.txt","FastText_All_StopwordsRemoval_test.txt")
end = time.time()
print("Time in s: %f s" %(end-start))


# In[ ]:




