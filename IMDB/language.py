import string
import torch

class Lang:

    def __init__(self):
        self.word2count = {}
        self.n_words = 0
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        return

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
        return

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        return



class LangEmbed:

    def __init__(self, embedding_size):
        self.word2count = {}
        self.n_words = 0
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        self.embedding_size = embedding_size
        self.embedding = {}
        self.mean = []
        self.std = []
        return

    def addWord_with_vector(self, word, v):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
            self.embedding[word] = v
        else:
            self.word2count[word] += 1
        return

    def get_mean_std(self):
        all_tensors = torch.zeros((len(self.embedding), self.embedding_size ))
        count = 0
        for k in self.embedding.keys():
            all_tensors[count,:] = self.embedding[k]
            count += 1
        self.mean = torch.mean(all_tensors, axis=0)
        self.std = torch.std(all_tensors, axis = 0)
        return

    def addWord_no_vector(self, word):
        if word not in self.word2index:
            v = torch.normal(self.mean,self.std)
            self.addWord_with_vector(word, v)
        else:
            self.word2count[word] += 1
        return

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord_no_vector(word)
        return
