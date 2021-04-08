import os
import pickle
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Embedding, InputLayer
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import tensorflow as tf
from keras.layers import LSTM
from keras.layers import Dropout, Bidirectional
from keras.models import load_model
from keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with open('textv2.txt', 'r', encoding='utf8') as f:
    text = f.read()

max_words = 10000
tokenizer = Tokenizer(num_words=max_words, lower=False, split=' ', char_level=False)
tokenizer.fit_on_texts([text])
pickle.dump(tokenizer, open('tokenizer.obj', 'bw'))
a = defaultdict(list)
sentences = tokenizer.texts_to_sequences(text.split('\n'))
del text
x_train = list()
y_train = list()
for sentence in sentences:
    if len(sentence) < 2:
        continue
    x_train.extend(sentence[:-1])
    y_train.extend(sentence[1:])
del sentences
print(len(x_train), len(y_train))
X = np.array(x_train)
Y = to_categorical(np.array(y_train))
print(len(Y))
del x_train
del y_train
model = Sequential()
model.add(Embedding(max_words, 256))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(max_words, activation='softmax'))
# simple early stopping
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
print(model.summary())
history = model.fit(X, Y, batch_size=128, epochs=50, callbacks=[es])
model.save('1_word.h5')
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()