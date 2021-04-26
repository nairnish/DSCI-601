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
import xgboost
from xgboost import XGBClassifier


class my_model():

    def fit(self, X, y):

        X = self.clean_data(X)
        required_text_features = ['new_text']
        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1,2)))])

        preprocessor = ColumnTransformer(
            transformers=[
                *[(feature_name, text_transformer, feature_name)
                  for feature_name in required_text_features]
            ]
        )

        log_reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', XGBClassifier())])
        self.clf = log_reg_pipe

        self.clf.fit(X, y)

        return

    def predict(self, X):

        X = self.clean_data(X)
        required_text_features = ['new_text']
        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1, 2)))])

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

        # Removal of White Spaces
        df['text'] = df['text'].apply(lambda x: x.strip())

        # Removal of Stopwords
        df1 = self.stopwords_removal(df)

        return df1

    def stopwords_removal(self, X):
        stopwords_list_bosnian = stopwords.words("slovene")
        stopwords_list_indonesian = stopwords.words("indonesian")
        stopwords_list_spanish = stopwords.words("spanish")
        stopwords_list_portuguese = stopwords.words("portuguese")
        stopwords_list_french = stopwords.words("french")
        stopwords_list_collated = [
            stopwords_list_bosnian + stopwords_list_indonesian + stopwords_list_spanish + stopwords_list_portuguese + stopwords_list_french]
        new_text = []
        for i in range(len(X)):
            tokens = nltk.word_tokenize(X['text'][i])
            filtered_text = [t for t in tokens if t not in stopwords_list_collated]
            new_text.append(" ".join(filtered_text))

        X['new_text'] = new_text
        return X