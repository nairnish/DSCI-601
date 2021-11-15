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
nltk.download('stopwords')
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

    '''
    Description - This function tries to fit the data in the LinearSVC model
    @:param - X is the independent feature in the dataset which is the textual data
    @:param - y is the dependent class label in the dataset which is the language variant label
    '''

    def fit(self, X, y, testX, testY):
        # Data cleaning
        X = self.clean_data(X)
        testX = self.clean_data(testX)

        vectorizer = TfidfVectorizer()
        trainX = vectorizer.fit_transform(X)
        X_test = vectorizer.transform(testX)

        # initialize model and define the space of the hyperparameters to
        # perform the grid-search over
        model = SVR()
        kernel = ["linear", "rbf", "sigmoid", "poly"]
        tolerance = [1e-3, 1e-4, 1e-5, 1e-6]
        C = [1, 1.5, 2, 2.5, 3]
        grid = dict(kernel=kernel, tol=tolerance, C=C)

        # initialize a cross-validation fold and perform a grid-search to
        # tune the hyperparameters
        print("[INFO] grid searching over the hyperparameters...")
        cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
                                  cv=cvFold, scoring="neg_mean_squared_error")
        searchResults = gridSearch.fit(trainX, y)
        # extract the best model and evaluate it
        print("[INFO] evaluating...")
        bestModel = searchResults.best_estimator_
        print("R2: {:.2f}".format(bestModel.score(X_test, testY)))

        # Only keeping required text features
        required_text_features = ['new_text']
        # Vectorizing text data using TF-IDF vectorizer
        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1,2)))])

        # Using column transformer for dimensionality reduction
        preprocessor = ColumnTransformer(
            transformers=[
                *[(feature_name, text_transformer, feature_name)
                  for feature_name in required_text_features]
            ]
        )

        # Building the pipeline for preprocessing and training
        log_reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', LinearSVC())])
        self.clf = log_reg_pipe

        # Fitting the data to the model
        self.clf.fit(X, y)

        return

    '''
    Description - This function tries to predict the y labels for the test dataset based on the trained model
    @:param - X is the independent feature in the dataset which is the textual data
    '''

    def predict(self, X):

        # Data cleaning
        X = self.clean_data(X)
        # Only keeping required text features
        required_text_features = ['new_text']
        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1, 2)))])

        preprocessor = ColumnTransformer(
            transformers=[
                *[(feature_name, text_transformer, feature_name)
                  for feature_name in required_text_features]
            ]
        )
        log_reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', LinearSVC())])

        # Predicting the language variant labels on the test data using the trained model
        predictions = self.clf.predict(X)

        return predictions

    '''
    Description - This function cleans the dataset and removes any unwanted text features which is not helpful in model training
    @:param - X is the independent feature in the dataset which is the textual data
    '''

    def clean_data(self, X):

        df = pd.DataFrame(X)

        # Removal of special characters
        spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                      "*", "+", ",", "-", ".", "/", ":", ";", "<",
                      "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                      "`", "{", "|", "}", "~", "–"]
        for char in spec_chars:
            df['text'] = df['text'].str.replace(char, ' ')


        # Removal of web-tags
        df['text'] = df['text'].str.replace('https?://\S+|www\.\S+', ' ')

        # Removal of White Spaces
        df['text'] = df['text'].apply(lambda x: x.strip())

        # Removal of Stopwords
        df1 = self.stopwords_removal(df)

        return df1

    '''
    Description - This function removes stopwords from the textual data
    @:param - X is the independent feature in the dataset which is the textual data
    '''

    def stopwords_removal(self, X):
        # Creating list of stopwords from all available language variants
        stopwords_list_bosnian = stopwords.words("slovene")
        stopwords_list_indonesian = stopwords.words("indonesian")
        stopwords_list_spanish = stopwords.words("spanish")
        stopwords_list_portuguese = stopwords.words("portuguese")
        stopwords_list_french = stopwords.words("french")
        # Collating the list of stopwords
        stopwords_list_collated = [stopwords_list_bosnian + stopwords_list_indonesian + stopwords_list_spanish + stopwords_list_portuguese + stopwords_list_french]
        new_text = []
        for i in range(len(X)):
            # tokenizing each text data
            tokens = nltk.word_tokenize(X['text'][i])
            # Removing stopwords
            filtered_text = [t for t in tokens if t not in stopwords_list_collated]
            new_text.append(" ".join(filtered_text))

        X['new_text'] = new_text
        return X
