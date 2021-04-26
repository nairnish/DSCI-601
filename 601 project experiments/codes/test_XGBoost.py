import time
import sys
import pandas as pd
from project_XGBoost import my_model
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import csv
from sklearn.cluster import KMeans

sys.path.insert(0, '..')
from Evaluation.my_evaluation import my_evaluation

def test(train_data, validation_data, test_data, unlabelled_test_data):
    clf = my_model()
    # clf = KMeans(n_clusters=14)
    y_train = train_data['label']
    X_train = train_data.drop(['label'], axis=1)

    y_validation = validation_data['label']
    X_validation = validation_data.drop(['label'], axis=1)

    y_test = test_data['label']
    X_test = test_data.drop(['label'], axis=1)

    X_un_test = unlabelled_test_data

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
    #                                                     random_state=42)
    # split_point = int(0.8 * len(y))
    # X_train = X.iloc[:split_point]
    # X_test = X.iloc[split_point:]
    # y_train = y.iloc[:split_point]
    # y_test = y.iloc[split_point:]

    # y_p = clf.fit_predict(X_train)

    # print("Classes:")
    # print([(y_train[i], y_p[i]) for i in range(len(y_train))])
    # print("Centroids:")
    # print(clf.cluster_centers_)
    # print("Inertia: %f" % clf.inertia_)
    #
    # # Transform test data to cluster-distance space
    # dists = clf.transform(X_test)
    # print("Testing data:")
    # print(dists)
    # print(clf.predict(X_test))

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    eval = my_evaluation(predictions, y_test)
    f1 = eval.f1(target=1)
    accuracy = eval.accuracy()
    precision = eval.precision(target=1)
    recall = eval.recall(target=1)

    print(classification_report(y_test, predictions))

    return f1,accuracy,precision,recall


if __name__ == "__main__":
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
    result = test(train_data, validation_data, test_data, unlabelled_test_data)
    print(result)
    # print("F1 score: %f" % result)
    runtime = (time.time() - start) / 60.0
    print(runtime)