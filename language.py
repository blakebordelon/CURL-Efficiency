import string

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
