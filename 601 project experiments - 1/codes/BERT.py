from wordcloud import WordCloud
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer
from transformers import TFBertModel
from sklearn.preprocessing import LabelEncoder
from os import path

plt.style.use('ggplot')

# Defining all our palette colours.
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
primary_black = "#202022"
primary_bgcolor = "#f4f0ea"

# primary_green = px.colors.qualitative.Plotly[2]
twitter_mask = np.array(Image.open('/601 project experiments - 1/mask.png'))

def bert_encode(data, maximum_length):
    input_ids = []
    attention_masks = []

    for text in data:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,

            return_attention_mask=True,
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return np.array(input_ids), np.array(attention_masks)


def create_model(bert_model):
    input_ids = tf.keras.Input(shape=(60,), dtype='int32')
    attention_masks = tf.keras.Input(shape=(60,), dtype='int32')

    output = bert_model([input_ids, attention_masks])
    output = output[1]
    output = tf.keras.layers.Dense(32, activation='relu')(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_learning_curves(history, arr):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    for idx in range(2):
        ax[idx].plot(history.history[arr[idx][0]])
        ax[idx].plot(history.history[arr[idx][1]])
        ax[idx].legend([arr[idx][0], arr[idx][1]],fontsize=18)
        ax[idx].set_xlabel('A ',fontsize=16)
        ax[idx].set_ylabel('B',fontsize=16)
        ax[idx].set_title(arr[idx][0] + ' X ' + arr[idx][1],fontsize=16)



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
    wc.to_file(path.join('/Users/nishantnair/DSCI-601/' + lang_variant + "_output.png"))

if __name__ == "__main__":
    train_data = pd.read_csv("../DSLCC4 datasets/DSL-TRAIN.txt", sep='\t', header=None, names=['text', 'label'])
    validation_data = pd.read_csv("../DSLCC4 datasets/DSL-DEV.txt", sep='\t', header=None, names=['text', 'label'])
    test_data = pd.read_csv("../DSLCC4 datasets/DSL-TEST-GOLD.txt", sep='\t', header=None, names=['text', 'label'])
    df = pd.concat([train_data, validation_data], ignore_index=True)
    req_labels = ['pt-BR', 'pt-PT', 'es-AR', 'es-ES', 'es-PE']

    result_df = df[df.label.isin(req_labels)]
    result_df = result_df.reset_index()
    result_df = result_df.drop(['index'], axis=1)

    result_test_df = test_data[test_data.label.isin(req_labels)]
    result_test_df = result_test_df.reset_index()
    result_test_df = result_test_df.drop(['index'], axis=1)

    # generate wordcloud
    for i in range(len(req_labels)):
        top_words(result_df, req_labels[i])

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
    fig.add_trace(go.Bar(
        x=['es-PE'],
        y=[balance_counts[2]],
        name='es-PE',
        text=[balance_counts[2]],
        textposition='auto',
        # marker_color=primary_grey
    ))
    fig.add_trace(go.Bar(
        x=['pt-BR'],
        y=[balance_counts[3]],
        name='pt-BR',
        text=[balance_counts[3]],
        textposition='auto',
        # marker_color=primary_grey
    ))
    fig.add_trace(go.Bar(
        x=['pt-PT'],
        y=[balance_counts[4]],
        name='pt-PT',
        text=[balance_counts[4]],
        textposition='auto',
        # marker_color=primary_grey
    ))
    fig.update_layout(
        title='<span style="font-size:32px; font-family:Times New Roman">Dataset distribution by target</span>'
    )
    fig.show()


    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    # label encode categorical labels
    le = LabelEncoder()
    le.fit(df['label'])
    df['label_encoded'] = le.transform(df['label'])

    texts = df['text']
    target = df['label_encoded']

    train_input_ids, train_attention_masks = bert_encode(texts, 60)
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    model = create_model(bert_model)
    model.summary()

    history = model.fit(
        [train_input_ids, train_attention_masks],
        target,
        validation_split=0.2,
        epochs=3,
        batch_size=10
    )

    plot_learning_curves(history, [['loss', 'val_loss'], ['accuracy', 'val_accuracy']])