import pandas as pd
from keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense

plt.style.use('ggplot')

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

    # vectorize text by turning it into seq of integers or into a vector
    # limit datset upto 50,000 words
    # set max num of words in each text at 250

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 500
    # This is fixed.
    EMBEDDING_DIM = 500
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

    # train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    # first layer is embedded layer - uses 100 lengths of vectors to represent word
    # spatial dropout performs variational dropout in nlp
    # lstm layer with 100 memory units
    # output layer - has 5 output values one for each language variant
    # softmax activation for multi-class classification
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 10
    batch_size = 64

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    accr = model.evaluate(X_test, Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

    plt.figure()
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

    test_data = test_data[test_data.label.isin(req_labels)]
    test_data = test_data.reset_index()
    test_data = test_data.drop(['index'], axis=1)

    new_text = ['Não foi ao Coliseu porque não gosta, declarou Elma. A namorada do futebolista, Irina Shayk, não esteve com ele na festa porque, conta ainda a irmã, "ela tem trabalho, tal como o meu irmão, que também já foi hoje para o estágio da seleção']
    new_texts = np.asarray(test_data['text'])
    seq = tokenizer.texts_to_sequences(new_texts.values)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    predictions = np.argmax(pred, axis=1)
    labels = ['pt-BR', 'pt-PT', 'es-AR', 'es-ES', 'es-PE']
    print(pred, labels[np.argmax(pred)])
