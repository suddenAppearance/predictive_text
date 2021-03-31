import os
import pickle
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as sess

import numpy as np
from tensorflow.python.keras.models import load_model


class Model:
    def __init__(self):
        print('Loading dictionary')
        self.tokenizer: Tokenizer = pickle.load(open('tokenizer.obj', 'rb'))
        self.maxWordsCount = 9997
        a = time.time()
        print('Loading model1')
        self.model1 = load_model('1_word.h5')
        print('Loading model2')
        self.model2 = load_model('2_word.h5')
        print('Loading model3')
        self.model3 = load_model('3_word.h5')
        print(time.time() - a)
        # models 4 and 5 are commented out to save memory and speed up the load can be uncommented if needed
        # self.model4 = load_model('model4')
        # self.model5 = load_model('model5')
        print('Models loaded. Launching...')

    def buildPhrase(self, data):
        if data == '' or data is None:
            return ''
        data = data.split()
        data = data[max(0, len(data) - 3):]
        i = 0
        while i < len(data):
            ind = self.tokenizer.word_index.get(data[i])
            if ind is None or ind > self.maxWordsCount:
                try:
                    data = data[i + 1:]
                    i = 0
                except IndexError:
                    data = []
            else:
                i += 1
        if os.environ['PT_LOGGING']:
            print(f'LOGS: list of last recognized words is {data}')
        l = len(data)
        if len(data) == 0:
            return ''
        if os.environ['PT_LOGGING']:
            print(f'LOGS: using {l} words prediction model')
        data = self.tokenizer.texts_to_sequences(data)[0]
        ans = ''
        if l == 1:
            ans = self.build1(data)
        if l == 2:
            ans = self.build2(data)
        if l == 3:
            ans = self.build3(data)
        # if len(data) == 4:
        #     return self.build4(data)
        # if len(data) == 5:
        #     return self.build5(data)
        # else:
        #     return ''
        if os.environ['PT_LOGGING']:
            print(f'LOGS: for {str(data)} array model predicted word: {ans}')
        return ans

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
        x = data
        inp = np.expand_dims(x, axis=0)

        pred = self.model3.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res = self.tokenizer.index_word[indx]  # дописываем строку

        return res

    def build2(self, data):
        x = data
        inp = np.expand_dims(x, axis=0)

        pred = self.model2.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res = self.tokenizer.index_word[indx]  # дописываем строку

        return res

    def build1(self, data):
        x = data
        inp = np.expand_dims(x, axis=0)

        pred = self.model1.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res = self.tokenizer.index_word[indx]  # дописываем строку

        return res


if __name__ == '__main__':
    model = Model()
    for i in range(1000):
        print(model.buildPhrase(input()))
