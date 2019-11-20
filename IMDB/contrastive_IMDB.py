import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import language
import timeit
import re
import string
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier

#FORMATTED_DATA_PATH = '../formatted_data/formatted_data.txt'
PATH0 = '../../aclImdb_v1.tar/aclImdb/movie_data/'
UNSUP_PATH = PATH0 + 'full_train_unsup.txt'
POS_TRAIN_PATH = PATH0 + 'full_train_pos.txt'
NEG_TRAIN_PATH = PATH0 + 'full_train_neg.txt'
POS_TEST_PATH = PATH0 + 'full_test_pos.txt'
NEG_TEST_PATH = PATH0 + 'full_test_pos.txt'

ALL_PATHS = [UNSUP_PATH, POS_TRAIN_PATH, NEG_TRAIN_PATH, POS_TEST_PATH, NEG_TEST_PATH]


def collect_formatted_strings(PATH):
    all_strings = []
    str_count = 0

    for line in open(PATH, 'r',  encoding = 'utf-8'):
        line = re.sub(r'[^\w ^\s]', '', line)
        line = line.lower()
        #line = re.sub(r'[A-Z]\\s', '', line)
        #print(line_formatted)

        tokens = line.split(' ')

        tokens = [t for t in tokens if t.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        #tokens = [word for word in tokens if len(word) > 1]
        separator = ' '
        #print(tokens)
        if len(tokens) > 0:
            all_strings.append(separator.join(tokens))
    return all_strings


def get_all_strs():
    collection = []
    for p in ALL_PATHS:
        collection.append(collect_formatted_strings(p))
    return collection


collection = get_all_strs()
[unsup_strs, pos_train_strs, neg_train_strs, pos_test_strs, neg_test_strs] = collection

train_strs = [pos_train_strs, neg_train_strs, pos_test_strs, neg_test_strs]

print("LEN unsup strs")
print(len(unsup_strs))

lang = language.Lang()

for c in collection:
    for s in c:
        #x = ' '.join(s)
        lang.addSentence(s)

print("done with language preprocessing")


n_words = lang.n_words
print("NUMBER OF WORDS")
print(n_words)


def get_random_string(all_strs, b):
    n_strs = len(all_strs)
    done = False
    while done == False:
        ind = np.random.randint(0,n_strs)
        if len(all_strs[ind].split(' ')) > b:
            done = True
    return all_strs[ind]

def get_substring(str, b):
    #str_arr = str.split(' ')
    str_arr = str.split(' ')
    length = len(str_arr)
    max_len = max(length, b)
    min_len = min(length, b)
    rand1 = np.random.randint(0, length - b + 1)
    return ' '.join(str_arr[rand1:rand1+b])

def str_to_tensor(str):
    str_arr = str.split(' ')
    inds = []
    for s in str_arr:
        if s in lang.word2index:
            inds.append(lang.word2index[s])
        else:
            lang.addWord(s)
            inds.append(lang.word2index[s])
    #ind_s = [lang.word2index[s] for s in str_arr]
    return torch.tensor(inds, dtype = torch.long)

def sample_tensor(all_strs, seq_length):
    x = torch.zeros((seq_length, 1))
    x[:,0] = str_to_tensor(get_substring(get_random_string(pos_train_strs, seq_length), seq_length)).type(torch.long)
    return x

def sample_contrastive(all_strs, seq_length, num_neg, block):

    str1 = get_random_string(all_strs, seq_length)
    X1 = torch.zeros((seq_length, block+1))
    X2 = torch.zeros((seq_length, block*num_neg))
    for b in range(block+1):
        X1[:,b] = str_to_tensor(get_substring(str1,seq_length))
    for i in range(num_neg):
        str_i = get_random_string(all_strs, seq_length)
        for b in range(block):
            X2[:,i*block+b] = str_to_tensor(get_substring(str_i, seq_length))
    return X1, X2

class RNN(nn.Module):

    def __init__(self, n_words, emb_dim, hidden_dim, seq_length, bidirectional=False, supervised = False):
        super(RNN, self).__init__()

        self.n_words = n_words
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.supervised = supervised
        self.seq_length = seq_length
        self.embedding = nn.Embedding(n_words, emb_dim)
        self.GRU = nn.GRU(emb_dim, hidden_dim)
        self.linear = nn.Linear(seq_length * hidden_dim, hidden_dim)
        self.attention = nn.Linear(emb_dim, 1)
        self.softmax = nn.Softmax()

        if bidirectional==True:
            self.GRU = nn.GRU(emb_dim, hidden_dim, bidirectional=True)
            self.linear = nn.Linear(2*seq_length * hidden_dim, hidden_dim)

        if self.supervised == True:
            self.linear2 = nn.Linear(hidden_dim, 2)
            self.sm = nn.LogSoftMax()

        return

    # consider x as as a vector of longs: seq_len x batch
    def forward(self, x):

        f,h = self.GRU(self.embedding(x))
        alpha = self.softmax(self.attention(self.embedding(x)))
        f_weighted = torch.zeros((f.shape[1], f.shape[2]))

        f_weighted = torch.bmm(f.permute(1,2,0), alpha.permute(1,0,2)).squeeze()
        return f_weighted



class TrainContrastive:

    def __init__(self, num_iter, lr, seq_length, num_neg, block):
        self.num_iter = num_iter
        self.lr = lr
        self.seq_length = seq_length
        self.num_neg = num_neg
        self.block = block

    def train_unsup(self, model, unsup_strs, train_strs):

        optimizer = optim.Adam(model.parameters(), lr = lr)
        trains = []
        tests = []
        for t in range(num_iter):
            start = timeit.default_timer()
            print("about to sample")
            X1, X2 = sample_contrastive(unsup_strs, self.seq_length, self.num_neg, self.block)
            print("sampled")
            print("forward pass")

            fX = model.forward(X1.type(torch.long))
            fX0 = fX[0,:]
            fX1 = fX[1:self.block+1,:]
            fX1_mean = fX1.mean(axis=0)
            fX2 = model.forward(X2.type(torch.long))

            loss = 0
            for k in range(self.num_neg):
                fXk_mean = fX2[k*self.block:(k+1)*self.block,:].mean(axis=0)

                loss += 1/self.num_neg * torch.log(1+torch.exp(torch.dot(fX0, fXk_mean) - torch.dot(fX0, fX1_mean)))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss.detach().numpy())
            if t % 100 == 0:
                train, test = self.train_sup(model, train_strs, 500)
                trains.append(train)
                tests.append(test)
                model.train()

        plt.plot(trains, label = 'train')
        plt.plot(tests, label = 'test')
        plt.xlabel('iterations in units 100')
        plt.ylabel('Downstream Classification Risk')
        plt.legend()
        plt.savefig('risk.pdf')
        plt.show()
        return

    def train_sup(self, model, train_strs , num_samples_each):
        pos_train_strs, neg_train_strs, pos_test_strs, neg_test_strs = train_strs
        model.eval()
        F = np.zeros((2*num_samples_each, model.hidden_dim))
        F_test = np.zeros((2*num_samples_each, model.hidden_dim))
        for i in range(num_samples_each):
            xp = sample_tensor(pos_train_strs, model.seq_length)
            F[i,:] = model.forward(xp.type(torch.long)).detach().numpy()
            xp = sample_tensor(pos_test_strs, model.seq_length)
            F_test[i,:] = model.forward(xp.type(torch.long)).detach().numpy()
        for i in range(num_samples_each):
            xm = sample_tensor(neg_train_strs, model.seq_length)
            F[num_samples_each+i,:] = model.forward(xm.type(torch.long)).detach().numpy()
            xm = sample_tensor(neg_test_strs, model.seq_length)
            F_test[num_samples_each+i,:] = model.forward(xm.type(torch.long)).detach().numpy()
        y = np.ones(2*num_samples_each)

        y[num_samples_each:len(y)] = -np.ones(num_samples_each)
        y_test = np.ones(2*num_samples_each)
        y_test[num_samples_each:len(y_test)] = - np.ones(num_samples_each)
        clf = SGDClassifier()
        clf.fit(F,y)

        train_acc = clf.score(F,y)
        test_acc = clf.score(F_test, y_test)
        print("SUPERVISED RESULTS!!!!!")
        print("train_acc = %lf" % train_acc)
        print("test acc = %lf" % test_acc)

        return train_acc, test_acc

print("about to start training")
num_iter = 20000
lr = 1e-4
seq_length = 25
num_neg =5
block = 3
emb_dim = 500
hidden_dim = 300

model = RNN(lang.n_words, emb_dim, hidden_dim, seq_length)
trainer = TrainContrastive(num_iter, lr, seq_length, num_neg, block)
trainer.train_unsup(model, unsup_strs, train_strs)
trainer.train_sup(model, train_strs, 1000)
