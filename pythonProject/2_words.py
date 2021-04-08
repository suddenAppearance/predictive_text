import json
import matplotlib.pyplot as plt
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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
with open('textv2.txt', 'r', encoding='utf8') as f:
    text = f.read()
for max_words in [20000]:
  for inp_words in [1, 2, 3]:
      epochs = 50
      tokenizer = Tokenizer(num_words=max_words, lower=False, split=' ', char_level=False)
      tokenizer.fit_on_texts([text])
      pickle.dump(tokenizer, open(f'tokenizer{max_words}.obj', 'bw'))
      a = defaultdict(list)
      sentences = tokenizer.texts_to_sequences(text.split('\n'))
      x_train = list()
      y_train = list()
      for sentence in sentences:
          if len(sentence) < inp_words + 1:
              continue
          for i in range(len(sentence)-inp_words):
              x_train.append(sentence[i:i+inp_words])
              y_train.append(sentence[i+inp_words])
      print(len(x_train), len(y_train))
      X = np.array(x_train)
      Y = to_categorical(np.array(y_train))
      print(len(Y))
      model = Sequential()
      model.add(Embedding(max_words, 256))
      model.add(LSTM(128, return_sequences=True))
      model.add(LSTM(128))
      model.add(Dense(max_words, activation='softmax'))

      model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
      print(model.summary())
      history = model.fit(X, Y, batch_size=128, epochs=epochs)
      del X
      del Y
      plt.plot(history.history['accuracy'])
      plt.plot(history.history['accuracy'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()
      # summarize history for loss
      plt.plot(history.history['loss'])
      plt.plot(history.history['loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()
      model.save(f'{inp_words}_word.h5')