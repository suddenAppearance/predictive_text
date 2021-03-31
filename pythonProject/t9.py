import os

from tensorflow.keras.preprocessing.text import Tokenizer
import pickle


class T9:
    tokenizer = pickle.load(open('tokenizer.obj', 'rb'))
    words = tokenizer.word_counts
    # print(sorted(tokenizer.word_counts.items(), key=lambda s: s[1]))
    words = sorted(words.items(), key=lambda s: s[1], reverse=True)

    def complete(self, pre_word):
        for word, count in self.words:
            if pre_word == word[:len(pre_word)]:
                ans = word[len(pre_word):]
                if os.environ['PT_LOGGING']:
                    print(f'LOGS: T9 completion for:{pre_word} is {word}, returning {ans}')
                return ans
        return ''


if __name__ == '__main__':
    T9().complete('ребя')
