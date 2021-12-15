import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from data_load import load_data
from feature_reduce import vectorizer_tfidf_unigram_bigram
from feature_reduce import column_transformer

'''
Plot history of the model

@param: history - model checkpoints
@return: None 
'''
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    return None

if __name__ == "__main__":
    train_path = '../../data/DSL-TRAIN.txt'
    val_path = '../../data/DSL-DEV.txt'
    test_path = '../../data/DSL-TEST-GOLD.txt'
    train_data = load_data(train_path)
    validation_data = load_data(val_path)
    test_data = load_data(test_path)
    df = pd.concat([train_data, validation_data], ignore_index=True)
    req_labels = ['pt-BR', 'pt-PT', 'es-AR', 'es-ES', 'es-PE']

    rslt_df = df[df.label.isin(req_labels)]
    rslt_df = rslt_df.reset_index()
    rslt_df = rslt_df.drop(['index'], axis=1)

    sentences = rslt_df['text'].values
    y = pd.get_dummies(rslt_df['label']).values

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

    X_train = column_transformer(sentences_train)
    X_test = column_transformer(sentences_test)

     
    input_dim = X_train.shape[1]  # Number of features

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train,
                        epochs=10,
                        verbose=True,
                        validation_data=(X_test, y_test),
                        batch_size=10)

    plot_history(history)

