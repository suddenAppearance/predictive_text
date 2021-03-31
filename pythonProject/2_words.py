import os
import pickle
from collections import defaultdict, Counter

import numpy as np

from tensorflow.keras.layers import Dense, Embedding, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.python.keras.layers import Dropout, Bidirectional
from tensorflow.python.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with open('textv2.txt', 'r', encoding='utf8') as f:
    text = f.read()

max_words = 9997
inp_words = 2
tokenizer = Tokenizer(num_words=max_words, lower=False, split=' ', char_level=False)
tokenizer.fit_on_texts([text])
pickle.dump(tokenizer, open('tokenizer.obj', 'bw'))
a = defaultdict(list)
sentences = tokenizer.texts_to_sequences(text.split('\n'))
x_train = list()
y_train = list()
for sentence in sentences:
    if len(sentence) < inp_words + 1:
        continue
    if len(x_train) != len(y_train):
        print(len(x_train))
        break
    for i in range(len(sentence)-inp_words):
        x_train.append(sentence[i:i+inp_words])
        y_train.append(sentence[i+inp_words])
print(len(x_train), len(y_train))
X = np.array(x_train)
Y = to_categorical(np.array(y_train))
print(len(Y))
model = Sequential()
model.add(Embedding(max_words, 256))
model.add(LSTM(128))
model.add(Dense(max_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
print(model.summary())
history = model.fit(X, Y, batch_size=64, epochs=50)
model.save('1_word.h5')
