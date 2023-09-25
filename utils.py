import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np


class NltkUtils:
    def __init__(self) -> None:
        self.stemmer = PorterStemmer()
    
    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def stem(self, word):
        return self.stemmer.stem(word.lower())
    
    def bagOfWords(self, tokenized_sentence, words):
        sentence_words = [self.stem(word) for word in tokenized_sentence]
        # initialize bag with 0 for each word
        bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in sentence_words: 
                bag[idx] = 1

        return bag
