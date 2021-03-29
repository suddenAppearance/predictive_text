from tensorflow.keras.preprocessing.text import Tokenizer


class T9:
    with open('Lepra_wall.txt', 'r', encoding='utf-8') as f:
        texts = f.read()
        texts = texts.replace('\ufeff', '')  # убираем первый невидимый символ

    with open('Lepra_wall.txt', 'r', encoding='utf-8') as f:
        texts2 = f.read()
        texts2 = texts.replace('\ufeff', '')  # убираем первый невидимый символ
    texts += texts2

    maxWordsCount = 10000
    tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                          lower=True, split=' ', char_level=False)
    tokenizer.fit_on_texts([texts])
    words = tokenizer.word_counts
    words = sorted(words.items(), key=lambda s: s[1], reverse=True)

    def complete(self, pre_word):
        for word, count in self.words:
            if pre_word == word[:len(pre_word)]:
                print(word)
                return word[len(pre_word):]
        return ''

if __name__ == '__main__':
    T9().complete('ребя')