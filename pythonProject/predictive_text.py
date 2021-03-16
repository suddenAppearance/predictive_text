from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np
from tensorflow.python.keras.models import load_model


class Model:
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
    inp_words = 5
    n = res.shape[0] - inp_words
    model1 = load_model('model1')
    model2 = load_model('model2')
    model3 = load_model('model3')
    model4 = load_model('model4')
    model5 = load_model('model5')

    def buildPhrase(self, data):
        if data == '' or data is None:
            return ''
        data = self.tokenizer.texts_to_sequences([data])[0]
        data = data[len(data) - 5:]
        if len(data) == 1:
            return self.build1(data)
        if len(data) == 2:
            return self.build2(data)
        if len(data) == 3:
            return self.build3(data)
        if len(data) == 4:
            return self.build4(data)
        if len(data) == 5:
            return self.build5(data)
        else:
            return ''

    def build5(self, data):
        x = data[:self.inp_words]
        inp = np.expand_dims(x, axis=0)

        pred = self.model5.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res = self.tokenizer.index_word[indx]  # дописываем строку

        return res

    def build4(self, data):
        x = data[:self.inp_words]
        inp = np.expand_dims(x, axis=0)

        pred = self.model4.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res = self.tokenizer.index_word[indx]  # дописываем строку

        return res

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
