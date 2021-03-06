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

'''
Description - implements the fit() and predict() methods of the model
@:param - train_data Training Dataset 
@:param - validation_data Validation Dataset 
@:param - test_data Test Dataset 
@:param - unlabelled_test_data Test Dataset without the y labels (language variant) 
'''

def test(train_data, validation_data, test_data, unlabelled_test_data):
    clf = my_model()
    # Getting only the y labels from training data
    y_train = train_data['label']
    # Getting only the independent input features in the training data
    X_train = train_data.drop(['label'], axis=1)

    y_validation = validation_data['label']
    X_validation = validation_data.drop(['label'], axis=1)

    y_test = test_data['label']
    # Getting only the independent input features in the test data
    X_test = test_data.drop(['label'], axis=1)

    X_un_test = unlabelled_test_data

    # fits the training data into the model
    clf.fit(X_train, y_train)
    # Getting the model predictions
    predictions = clf.predict(X_test)
    # Generating evaluation (F1, accuracy, precision, recall) on the predicted data and ground truth labels
    eval = my_evaluation(predictions, y_test)
    f1 = eval.f1(target=1)
    accuracy = eval.accuracy()
    precision = eval.precision(target=1)
    recall = eval.recall(target=1)

    print(classification_report(y_test, predictions))

    return f1,accuracy,precision,recall


if __name__ == "__main__":
    # Recording the start time of the process
    start = time.time()
    # Load train data
    train_data = pd.read_csv("../DSLCC4 datasets/DSL-TRAIN.txt", sep='\t', header=None, names=['text', 'label'])
    # Load validation data
    validation_data = pd.read_csv("../DSLCC4 datasets/DSL-DEV.txt", sep='\t', header=None, names=['text', 'label'])
    # Load test data
    test_data = pd.read_csv("../DSLCC4 datasets/DSL-TEST-GOLD.txt", sep='\t', header=None, names=['text', 'label'])
    # Load unlabelled test data
    unlabelled_test_data = pd.read_csv("../DSLCC4 datasets/DSL-TEST-UNLABELLED.txt", sep='\t', header=None, names=['text'])
    # Preliminary data cleaning - Replace missing values with empty strings
    train_data = train_data.fillna("")
    validation_data = validation_data.fillna("")
    test_data = test_data.fillna("")
    unlabelled_test_data = unlabelled_test_data.fillna("")

    # # Filtering portuguese data
    # new_train_data = train_data[(train_data.label == "pt-BR") | (train_data.label == "pt-PT")]
    # new_validation_data = validation_data[(validation_data.label == "pt-BR") | (validation_data.label == "pt-PT")]
    # new_test_data = test_data[(test_data.label == "pt-BR") | (test_data.label == "pt-PT")]
    #
    # # Filtering Spanish data
    # new_train_data = train_data[(train_data.label == "es-AR") | (train_data.label == "es-ES") | (train_data.label == "es-PE")]
    # new_validation_data = validation_data[(validation_data.label == "es-AR") | (validation_data.label == "es-ES") | (validation_data.label == "es-PE")]
    # new_test_data = test_data[(test_data.label == "es-AR") | (test_data.label == "es-ES") | (test_data.label == "es-PE")]
    #
    # result = test(new_train_data, new_validation_data, new_test_data, unlabelled_test_data)

    # Results of the prediction in the form {F1, Accuracy, Precision, Recall} is generated by testing the classifier
    result = test(train_data, validation_data, test_data, unlabelled_test_data)
    print(result)
    # print("F1 score: %f" % result)
    # Recording the total time taken for the model training and prediction process
    runtime = (time.time() - start) / 60.0
    print(runtime)