import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import re
import csv
from sklearn.feature_extraction.text import CountVectorizer
import timeit
import matplotlib.pyplot as plt

OLDPATH = '../aclImdb_v1.tar/aclImdb/train/unsup/'
PATH = '../formatted_data/'

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def index_from_str(sentence, lang):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensor_from_str(sentence, lang):
    indexes = index_from_str(sentence, lang)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)



review_lang = Lang('review_lang')


def collect_all_data(num_data):

    all_data = []
    for i in range(num_data):
        data_path = OLDPATH + str(i) +'_0.txt'
        data = []
        print(data_path)
        try:
            for line in open(data_path):
                data.append(line.strip())
            all_data.append(data)
        except:
            print("skipping a data point")

    return all_data




def save_formatted(formatted_data):

    with open(PATH + 'formatted_data.txt', 'w') as f:
        csvwriter = csv.writer(f)

        for r in formatted_data:
            csvwriter.writerow(r)
    return

def preprocess(reviews):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews

num_data = 5000
data = collect_all_data(num_data)
formatted_data = [preprocess(r) for r in data]
save_formatted(formatted_data)


for s in formatted_data:
    review_lang.addSentence(s[0])

all_tensor_data = [tensor_from_str(s[0], review_lang) for s in formatted_data]

input_dim = all_tensor_data[0].shape[0]

def sample_similar_words(neg_samples):

    rand_indx = np.random.randint(0, len(all_tensor_data))
    sentence = all_tensor_data[rand_indx]
    s_len = sentence.shape[0]
    rands = np.random.randint(0, s_len, 2)
    x = sentence[rands[0]]
    xp = sentence[rands[1]]
    x.type(torch.long)
    xp.type(torch.long)
    rand_min = np.random.randint(0, len(all_tensor_data))
    sentence = all_tensor_data[rand_min]
    xm = torch.zeros((neg_samples, x.shape[0]))
    for k in range(neg_samples):
        xm[k,:] = sentence[np.random.randint(0, len(sentence))]
    xm.type(torch.long)
    xf = torch.zeros(review_lang.n_words)
    xfp = torch.zeros(review_lang.n_words)
    xfm = torch.zeros((neg_samples, review_lang.n_words))

    xf[x] = 1
    xfp[xp] = 1
    xfm[:,xm[:,0].type(torch.long)] = torch.ones(neg_samples)

    return xf,xfp,xfm

input_dim = review_lang.n_words
class CURL_Embedding(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super(CURL_Embedding, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #self.embedding = nn.Embedding(input_dim , hidden_dim)
        self.first = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace = False)
        self.second = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        #return self.relu(self.second(self.relu(self.first(self.embedding(x)))))
        return self.relu(self.second(self.relu(self.first(x))))

print(all_tensor_data[0].shape)
hidden_dim = 20
lr = 1e-3
iter = 200
neg_samples = 10
batch_size = 100
model = CURL_Embedding(input_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr = lr)
all_losses = []
for t in range(iter):
    print("t = %d" % t)

    x_b = torch.zeros((batch_size, input_dim))
    xp_b = torch.zeros((batch_size, input_dim))
    xm_b = torch.zeros((batch_size, neg_samples,input_dim))
    for b in range(batch_size):
        x_b[b,:],xp_b[b,:], xm_b[b,:,:] = sample_similar_words(neg_samples)


    with torch.autograd.set_detect_anomaly(False):

        #x = torch.tensor(0).type(torch.long)
        start_forward = timeit.default_timer()
        forwardx = model.forward(x_b).squeeze()
        forwardxp = model.forward(xp_b).squeeze()
        loss = 0.0

        for k in range(neg_samples):
            forward_k = model.forward(xm_b[:,k,:]).squeeze()
            for b in range(batch_size):
                loss += 1/batch_size * torch.log(1 + torch.exp( torch.dot(forwardx[b,:], forward_k[b,:]) - torch.dot(forwardx[b,:], forwardxp[b,:])  )).sum()

        end_forward = timeit.default_timer()

        loss.backward()

        print("loss: %lf" % loss)
        end_back = timeit.default_timer()
        optimizer.step()
        optimizer.zero_grad()

        print("foward pass time: %lf" % (end_forward-start_forward))
        print("back pass time: %lf" % (end_back - end_forward))
    all_losses.append(loss)

plt.plot(all_losses)
plt.xlabel('iteration')
plt.ylabel('Contrastive Loss')
plt.savefig('contrastive_loss.pdf')
plt.show()


# To do: train a linear classifier on the output: Need to load labeled data

reviews_train = []
for line in open('../aclImdb_v1.tar/aclImdb/movie_data/full_train.txt', 'r'):
    reviews_train.append(line.strip())

reviews_train = preprocess(reviews_train)
target = [1 if i < 12500 else 0 for i in range(25000)]
