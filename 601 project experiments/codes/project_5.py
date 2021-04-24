import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
import nltk as nltk
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC, LinearSVC
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import re
from nltk import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from collections import OrderedDict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from gensim.parsing.preprocessing import remove_stopwords


class my_model():

    def fit_hint(self, X, y):
        # do not exceed 29 mins
        X = self.clean_data(X)
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
        # XX = self.preprocessor.fit_transform(X["description"])
        XX = self.preprocessor.fit_transform(X["combined_text"])
        self.clf = SGDClassifier(class_weight="balanced")
        self.clf.fit(XX, y)
        return

    def fit(self, X, y):
        # combine X and y in 1 data
        # y1 = pd.DataFrame(y)
        # X1 = pd.DataFrame(X)
        # X1.index = y1.index
        # data = pd.concat([X1, y1], axis=1)
        # # data = X.append(y)
        # # data = pd.concat([X + y.to_frame().T])
        #
        # # oversampling records with fraudulent records
        # data_1f = data[data.fraudulent == 1]
        # original_data = data.copy()
        # data = pd.concat([data] + [data_1f] * 3, axis=0)

        # do not exceed 29 mins
        X = self.clean_data(X)
        required_text_features = ['text']
        # binary_transformer = Pipeline(steps=[('label', OneHotEncoder(handle_unknown='ignore'))])
        # cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1,2)))])
        # text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False))])

        preprocessor = ColumnTransformer(
            transformers=[
                # ('bin', binary_transformer, required_binary_features),
                # ('cat', cat_transformer, required_cat_features),
                *[(feature_name, text_transformer, feature_name)
                  for feature_name in required_text_features]
            ]
        )

        log_reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', LogisticRegression())])
        self.clf = log_reg_pipe

        self.clf.fit(X, y)

        return

    def predict_hint(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        # XX = self.preprocessor.transform(X["description"])
        X = self.clean_data(X)
        XX = self.preprocessor.transform(X['combined_text'])
        predictions = self.clf.predict(XX)
        return predictions

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X = self.clean_data(X)
        required_text_features = ['text']
        # binary_transformer = Pipeline(steps=[('label', OneHotEncoder(handle_unknown='ignore'))])
        # cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1, 2)))])
        # text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False))])

        preprocessor = ColumnTransformer(
            transformers=[
                # ('bin', binary_transformer, required_binary_features),
                # ('cat', cat_transformer, required_cat_features),
                *[(feature_name, text_transformer, feature_name)
                  for feature_name in required_text_features]
            ]
        )
        # log_reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
        #                                ('classifier', LogisticRegression())])
        # self.clf = log_reg_pipe
        predictions = self.clf.predict(X)
        return predictions

    def clean_data(self, X):

        df = pd.DataFrame(X)

        spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                      "*", "+", ",", "-", ".", "/", ":", ";", "<",
                      "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                      "`", "{", "|", "}", "~", "–"]
        for char in spec_chars:
            df['text'] = df['text'].str.replace(char, ' ')

        df['text'] = df['text'].str.replace('https?://\S+|www\.\S+', ' ')

        return df