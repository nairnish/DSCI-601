# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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

    # truncate and pad input seq so that they are of the same length for modeling
    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 250
    # This is fixed.
    EMBEDDING_DIM = 100
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(rslt_df['text'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # truncate and pad input seq so that they are of the same length for modeling
    X = tokenizer.texts_to_sequences(rslt_df['text'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)

    # label encode categorical labels
    Y = pd.get_dummies(rslt_df['label']).values
    print('Shape of label tensor:', Y.shape)

    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    # create the model
    # embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))