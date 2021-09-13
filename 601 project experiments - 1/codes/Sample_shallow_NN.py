import time
import sys
import pandas as pd
# from project_5 import my_model
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from tensorflow import keras
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from keras.models import Sequential
from keras import layers
from sklearn import metrics
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import model_selection
import csv
import tensorflow as tf
from keras.backend import clear_session
import os


def create_model_architecture(X_train, y_train, X_test, y_test):

    input_dim = X_train.shape[1]  # Number of features

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, epochs = 100, verbose = False, validation_data = (X_test, y_test),batch_size = 10)

    clear_session()

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))



    # # create input layer
    # input_layer = layers.Input((input_size,), sparse=True)
    #
    # # create hidden layer
    # hidden_layer = layers.Dense(10, activation="relu")(input_layer)
    #
    # # create output layer
    # output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)
    #
    # classifier = models.Model(inputs=input_layer, outputs=output_layer)
    # classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start = time.time()
    # Load data
    train_data = pd.read_csv("../DSLCC4 datasets/DSL-TRAIN.txt", sep='\t', header=None, names=['text', 'label'])
    validation_data = pd.read_csv("../DSLCC4 datasets/DSL-DEV.txt", sep='\t', header=None, names=['text', 'label'])
    test_data = pd.read_csv("../DSLCC4 datasets/DSL-TEST-GOLD.txt", sep='\t', header=None, names=['text', 'label'])
    unlabelled_test_data = pd.read_csv("../DSLCC4 datasets/DSL-TEST-UNLABELLED.txt", sep='\t', header=None, names=['text'])
    # Replace missing values with empty strings
    train_data = train_data.fillna("")
    validation_data = validation_data.fillna("")
    test_data = test_data.fillna("")
    unlabelled_test_data = unlabelled_test_data.fillna("")

    # split the dataset into training and validation datasets
    # train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_data['text'], train_data['label'])

    y_train = train_data.drop(['text'], axis=1).squeeze()
    X_train = train_data.drop(['label'], axis=1).squeeze()

    y_test = test_data.drop(['text'], axis=1).squeeze()
    X_test = test_data.drop(['label'], axis=1).squeeze()

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(y_train)
    valid_y = encoder.fit_transform(y_test)

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
    tfidf_vect.fit(train_data['text'])
    xtrain_tfidf = tfidf_vect.transform(X_train).toarray()
    xvalid_tfidf = tfidf_vect.transform(X_test).toarray()

    # create a tokenizer
    # token = text.Tokenizer()
    # token.fit_on_texts(train_data['text'])
    # word_index = token.word_index

    # convert text to sequence of tokens and pad them to ensure equal length vectors
    # train_seq_x = sequence.pad_sequences(token.texts_to_sequences(X_train), maxlen=70)
    # valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(X_test), maxlen=70)

    # classifier = create_model_architecture(xtrain_tfidf.shape[1])
    classifier = create_model_architecture(xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
    # accuracy = train_model(classifier, xtrain_tfidf, train_y, xvalid_tfidf, is_neural_net=True)
    # print("NN, Ngram Level TF IDF Vectors", accuracy)

    runtime = (time.time() - start) / 60.0
    print(runtime)





    # result = test(train_data, validation_data, test_data, unlabelled_test_data)
    # print(result)
    # # print("F1 score: %f" % result)
    # runtime = (time.time() - start) / 60.0
    # print(runtime)