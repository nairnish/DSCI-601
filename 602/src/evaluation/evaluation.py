class evaluation():

'''
Import statements here.
'''
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def sklearn_eval_matrices(y_true, y_pred):
    print('Confusion Matrix: ',confusion_matrix(y_true, y_pred))
    print('Accuracy from sklearn: ', accuracy_score(y_true, y_pred))
    print('Precision from sklearn: ', precision_score(y_true, y_pred))
    print('Recall from sklearn: ', recall_score(y_true, y_pred))
    print('AOC from sklearn: ', roc_auc_score(y_true, y_pred))
    print('F1 from sklearn: ', f1_score(y_true, y_pred))

def skleran_class_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))

