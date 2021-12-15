import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from wordcloud import WordCloud
import numpy as np
from PIL import Image
from os import path
import plotly.graph_objects as go
# import tensorflow.python.keras.backend as K
# sess = K.get_session()
import tensorflow as tf

plt.style.use('ggplot')

# Defining all our palette colours.
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
primary_black = "#202022"
primary_bgcolor = "#f4f0ea"

# primary_green = px.colors.qualitative.Plotly[2]
twitter_mask = np.array(Image.open('/content/mask.png'))

def top_words(df, lang_variant):
    wc = WordCloud(
        background_color='white',
        max_words=200,
        mask=twitter_mask,
    )
    wc.generate(' '.join(text for text in df.loc[df['label'] == lang_variant, 'text']))
    plt.figure(figsize=(18, 10))
    plt.title('Top words for ' + lang_variant + ' language variant',
              fontdict={'size': 22, 'verticalalignment': 'bottom'})
    plt.imshow(wc)
    plt.axis("off")
    plt.show()

    # save wordcloud
    # wc.to_file(path.join('/content/' + lang_variant + "_output.png"))

def plot_histogram(result_df):
    result_df['text_len'] = result_df['text'].astype(str).apply(len)
    result_df['word_count'] = result_df['text'].apply(lambda x: len(str(x).split()))
    result_df.hist(column='text_len',
                   grid=False,
                   figsize=(10, 4),
                   legend=True,
                   bins=500,
                   orientation='vertical',
                   color='#A0E8AF')

    result_df.hist(column='word_count',
                   grid=False,
                   figsize=(10, 4),
                   legend=True,
                   bins=500,
                   orientation='vertical',
                   color='#FFCF56')

def plot_classDistribution(result_df):
    # generate bar chart to check class imbalance
    balance_counts = result_df.groupby('label')['label'].agg('count').values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['es-AR'],
        y=[balance_counts[0]],
        name='es-AR',
        text=[balance_counts[0]],
        textposition='auto',
        # marker_color=primary_blue
    ))
    fig.add_trace(go.Bar(
        x=['es-ES'],
        y=[balance_counts[1]],
        name='es-ES',
        text=[balance_counts[1]],
        textposition='auto',
        # marker_color=primary_grey
    ))
    # fig.add_trace(go.Bar(
    #     x=['es-PE'],
    #     y=[balance_counts[2]],
    #     name='es-PE',
    #     text=[balance_counts[2]],
    #     textposition='auto',
    #     # marker_color=primary_grey
    # ))
    fig.add_trace(go.Bar(
        x=['pt-BR'],
        y=[balance_counts[2]],
        name='pt-BR',
        text=[balance_counts[2]],
        textposition='auto',
        # marker_color=primary_grey
    ))
    fig.add_trace(go.Bar(
        x=['pt-PT'],
        y=[balance_counts[3]],
        name='pt-PT',
        text=[balance_counts[3]],
        textposition='auto',
        # marker_color=primary_grey
    ))
    fig.update_layout(
        title='<span style="font-size:32px; font-family:Times New Roman">Dataset distribution by target</span>'
    )
    fig.show()


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

if __name__ == "__main__":
    train_data = pd.read_csv("/content/DSL-DEV.txt", sep='\t', header=None, names=['text', 'label'])
    validation_data = pd.read_csv("/content/DSL-DEV.txt", sep='\t', header=None, names=['text', 'label'])
    test_data = pd.read_csv("/content/DSL-TEST-GOLD.txt", sep='\t', header=None, names=['text', 'label'])
    df = pd.concat([train_data, validation_data], ignore_index=True)
    req_labels = ['pt-BR', 'pt-PT', 'es-AR', 'es-ES']
    # req_labels = ['fr-CA', 'fr-FR', 'fa-AF', 'fa-IR']

    rslt_df = df[df.label.isin(req_labels)]
    rslt_df = rslt_df.reset_index()
    rslt_df = rslt_df.drop(['index'], axis=1)

    # generate wordcloud
    for i in range(len(req_labels)):
        top_words(rslt_df, req_labels[i])

    # generate histogram
    plot_histogram(rslt_df)

    # generate bar chart
    plot_classDistribution(rslt_df)

    sentences = rslt_df['text'].values
    y = pd.get_dummies(rslt_df['label']).values

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train).todense()
    X_test = vectorizer.transform(sentences_test).todense()

    input_dim = X_train.shape[1]  # Number of features

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    # model.add(layers.Dense(5, activation='softmax'))
    model.add(layers.Dense(4, activation='softmax'))
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

    print()


