import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as sess

import numpy as np
from tensorflow.python.keras.models import load_model


class Model:
    def __init__(self):
        with open('Lepra_wall.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
            texts = texts.replace('\ufeff', '')  # убираем первый невидимый символ

        with open('LIFE_wall.txt', 'r', encoding='utf-8') as f:
            texts2 = f.read()
            texts2 = texts.replace('\ufeff', '')  # убираем первый невидимый символ

        texts += texts2
        maxWordsCount = 1000
        tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                              lower=True, split=' ', char_level=False)
        tokenizer.fit_on_texts([texts])
        self.tokenizer = tokenizer
        self.inp_words = 5
        print('Loading model1')
        self.model1 = load_model('model1')
        print('Loading model2')
        self.model2 = load_model('model2')
        print('Loading model3')
        self.model3 = load_model('model3')
        # models 4 and 5 are commented out to save memory and speed up the load can be uncommented if needed
        # self.model4 = load_model('model4')
        # self.model5 = load_model('model5')
        print('Models loaded. Launching...')

    def buildPhrase(self, data):
        if data == '' or data is None:
            return ''
        data = self.tokenizer.texts_to_sequences([data])[0]
        data = data[len(data) - 3:]
        if len(data) == 1:
            return self.build1(data)
        if len(data) == 2:
            return self.build2(data)
        if len(data) == 3:
            return self.build3(data)
        # if len(data) == 4:
        #     return self.build4(data)
        # if len(data) == 5:
        #     return self.build5(data)
        # else:
        #     return ''

    # def build5(self, data):
    #     x = data[:self.inp_words]
    #     inp = np.expand_dims(x, axis=0)
    #
    #     pred = self.model5.predict(inp)
    #     indx = pred.argmax(axis=1)[0]
    #     data.append(indx)
    #
    #     res = self.tokenizer.index_word[indx]  # дописываем строку
    #
    #     return res
    #
    # def build4(self, data):
    #     x = data[:self.inp_words]
    #     inp = np.expand_dims(x, axis=0)
    #
    #     pred = self.model4.predict(inp)
    #     indx = pred.argmax(axis=1)[0]
    #     data.append(indx)
    #
    #     res = self.tokenizer.index_word[indx]  # дописываем строку
    #
    #     return res

    def build3(self, data):
        x = data[:self.inp_words]
        inp = np.expand_dims(x, axis=0)

        pred = self.model3.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res = self.tokenizer.index_word[indx]  # дописываем строку

        return res

    def build2(self, data):
        x = data[:self.inp_words]
        inp = np.expand_dims(x, axis=0)

        pred = self.model2.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res = self.tokenizer.index_word[indx]  # дописываем строку

        return res

    def build1(self, data):
        x = data[:self.inp_words]
        inp = np.expand_dims(x, axis=0)

        pred = self.model1.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res = self.tokenizer.index_word[indx]  # дописываем строку

        return res
