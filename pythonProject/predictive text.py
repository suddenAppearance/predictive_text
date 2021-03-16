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


    def buildPhrase(self, texts, str_len=1):
        res = texts
        data = self.tokenizer.texts_to_sequences([texts])[0]
        for i in range(str_len):
            # x = to_categorical(data[i: i + inp_words], num_classes=maxWordsCount)  # преобразуем в One-Hot-encoding
            # inp = x.reshape(1, inp_words, maxWordsCount)
            x = data[i: i + self.inp_words]
            inp = np.expand_dims(x, axis=0)

            pred = self.model5.predict(inp)
            indx = pred.argmax(axis=1)[0]
            data.append(indx)

            res += " " + self.tokenizer.index_word[indx]  # дописываем строку

        return res


model = Model()
for i in range(2):
    print(model.buildPhrase("Лучше не стоит этого"))
