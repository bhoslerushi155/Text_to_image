import os
import io
import numpy as np


class Glove(object):

    def __init__(self):
        self.encoding = {}
        self.encoding_dim = 100

    def load_glove(self , path):
        path += "/glove.6B.100d.txt"
        encd = {}
        f = io.open(path , mode='rt' , encoding='utf8')
        for line in f:
            words = line.strip().split()
            word = words[0]
            encode = np.array(words[1:] , dtype=np.float32)
            encd[word] = encode
        f.close()
        self.encoding = encd

    def encode_text(self , text):
        words = list()
        for w in text.split(' '):
            words.append(w.lower())
        length = len(words)
        enc = np.zeros(shape=(self.encoding_dim , length))
        vector = np.zeros(shape=(self.encoding_dim , ))
        vector[:] = np.sum(enc , axis=1)
        return vector
