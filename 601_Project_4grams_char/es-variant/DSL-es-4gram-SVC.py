import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

start = time.time()
# Load train data
train_data = pd.read_csv('/Users/neha/Documents/Semester_2/601TrailRuns/DSCI-601-DSL/dslcc4/DSL-TRAIN.txt', delimiter = "\t", header = None)

#renaming column name.
train_data = train_data.rename(columns = {0 : "text"})
train_data = train_data.rename(columns = {1 : "lang_variant"})

#Training the model just for spanish variant.
spanish_var = ['es-AR', 'es-ES', 'es-PE']

#Training the model just for spanish variant. Filtering out the spaish variant
train_data = train_data[train_data['lang_variant'].isin(spanish_var)]

#Removal of special characters : cleaning of data.
spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
              "*", "+", ",", "-", ".", "/", ":", ";", "<",
              "=", ">", "?", "@", "[", "\\", "]", "^", "_",
              "`", "{", "|", "}", "~", "–"]
for char in spec_chars:
    train_data['text'] = train_data['text'].str.replace(char, ' ')
train_data['text'] = train_data['text'].str.replace('https?://\S+|www\.\S+', ' ')

#Breaking the data columns into dependent and independent variable.
y_train = train_data["lang_variant"]
X_train = train_data["text"]

#vectorization
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(4,4))
X_train = vectorizer.fit_transform(X_train)

#Work on test data. It should be same as the other train data
test_data = pd.read_csv('/Users/neha/Documents/Semester_2/601TrailRuns/DSCI-601-DSL/dslcc4/DSL-TEST-GOLD.txt', delimiter = "\t", header = None)

#renaming column name.
test_data = test_data.rename(columns = {0 : "text"})
test_data = test_data.rename(columns = {1 : "lang_variant"})

#test dataset containing just spanish variant.
test_data = test_data[test_data['lang_variant'].isin(spanish_var)]

y_test = test_data["lang_variant"]
X_test = test_data["text"]

#Removal of special characters : cleaning of data.
spec_chars_test = ["!", '"', "#", "%", "&", "'", "(", ")",
              "*", "+", ",", "-", ".", "/", ":", ";", "<",
              "=", ">", "?", "@", "[", "\\", "]", "^", "_",
              "`", "{", "|", "}", "~", "–"]
for char in spec_chars_test:
    test_data['text'] = test_data['text'].str.replace(char, ' ')
test_data['text'] = test_data['text'].str.replace('https?://\S+|www\.\S+', ' ')


#vectorization
X_test = vectorizer.transform(X_test)

#Model building
# 1. SVC
clf = SVC()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

#Evaluation
print(classification_report(predictions,y_test))
print(f1_score(y_test, predictions, average='macro'))
print(accuracy_score(y_test,predictions))

model_runtime = (time.time() - start)
print(model_runtime)