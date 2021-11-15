# -*- coding: utf-8 -*-
"""LSTMForSequenceClassification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yOs5CYVrQc1JT7Gq2psiCG4bLpwZmYjL
"""

from google.colab import drive
drive.mount('/content/drive')
from data_load import load_data
import numpy as np
import pandas as pd
import torch
import time
import datetime
import random

train_path = 'MyDrive/DSLCCv4.0_1/DSL-TRAIN.txt'
dev_path = 'MyDrive/DSLCCv4.0_1/DSL-DEV.txt'
test_path = 'MyDrive/DSLCCv4.0_1/DSL-TEST-GOLD.txt'


df_train = preprocess(train_path)
df_test = preprocess(test_path)
df_val = preprocess(dev_path)

df_train = df_train[df_train["language"].isin(["pt-BR","pt-PT"])]
df_test =  df_test[df_test["language"].isin(["pt-BR","pt-PT"])]
df_val = df_val[df_val["language"].isin(["pt-BR","pt-PT"])]

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
## Plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
# Others
import nltk
import string
from nltk.corpus import stopwords

from sklearn.manifold import TSNE

### Create sequence
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df_train['text'])
sequences = tokenizer.texts_to_sequences(df_train['text'])
data = pad_sequences(sequences, maxlen=50)

map1 = {'pt-BR':0,'pt-PT':1}
labels = df_train["language"].map(map1)

map1 = {'pt-BR':0,'pt-PT':1}
labels_test = df_test["language"].map(map1)

model = Sequential()
model.add(Embedding(20000, 100, input_length=50))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## Fit the model
history = model.fit(data, np.array(labels), validation_split=0.4, epochs=10)

#Visualize the word embeddings
word_embds = model.layers[0].get_weights()

word_list = []
for word, i in tokenizer.word_index.items():
    word_list.append(word)

sequences = tokenizer.texts_to_sequences(df_test['text'])
data = pad_sequences(sequences, maxlen=50)
accr = model.evaluate(data,labels_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

