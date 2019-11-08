import string

class Lang:

    def __init__(self):
        self.word_counts = {}
        self.n_words = 0
        self.word_index = {}
        self.index_to_word = []
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
