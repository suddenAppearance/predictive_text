import os

import numpy as np

from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.python.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('Lepra_wall.txt', 'r', encoding='utf-8') as f:
    texts = f.read()
    texts = texts.replace('\ufeff', '')  # убираем первый невидимый символ

with open('LIFE_wall.txt', 'r', encoding='utf-8') as f:
    texts2 = f.read()
    texts2 = texts.replace('\ufeff', '')  # убираем первый невидимый символ
texts += texts2
print(len(texts))
maxWordsCount = 1000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                      lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts([texts])

dist = list(tokenizer.word_counts.items())
print(dist[:10])

data = tokenizer.texts_to_sequences([texts])
# res = to_categorical(data[0], num_classes=maxWordsCount)
# print(res.shape)

res = np.array(data[0])
print(len(res))
inp_words = 4
n = res.shape[0] - inp_words

X = np.array([res[i:i + inp_words] for i in range(n)])
Y = to_categorical(res[inp_words:], num_classes=maxWordsCount)
print(X)
print(Y)
model = Sequential()
model.add(Embedding(maxWordsCount, 256, input_length=inp_words))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(maxWordsCount, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X, Y, batch_size=64, epochs=50)
model.save('model4')
model = load_model('model4')


def buildPhrase(texts,):
    res = texts
    data = tokenizer.texts_to_sequences([texts])[0]

    # x = to_categorical(data[i: i + inp_words], num_classes=maxWordsCount)  # преобразуем в One-Hot-encoding
    # inp = x.reshape(1, inp_words, maxWordsCount)
    x = data[:inp_words]
    inp = np.expand_dims(x, axis=0)

    pred = model.predict(inp)
    indx = pred.argmax(axis=1)[0]  # this need to be rewritten whether u want 1 or more suggestions
    data.append(indx)

    return tokenizer.index_word[indx]


res = buildPhrase("люди сорок пять лет делали")
print(res)
