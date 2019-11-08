import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import language


FORMATTED_DATA_PATH = '../formatted_data/formatted_data.txt'

def collect_formatted_strings():
    all_strings = []
    for line in open('../formatted_data/formatted_data.txt'):
        all_strings.append(line.strip())

    return all_strings



strs = collect_formatted_strings()

lang = Lang()
for s in strs:
    lang.addSentence(s)

def convert_to_one_hot(strs, lang):
    all_tensors = []
    for s in strs:
        sentence_length = len(s.split(' '))
        tensor_s  = torch.zeros((sentence_length, lang.n_words))
        count = 0
        for w in s.split(' '):
            ind = lang.word2index[w]
            tensor_s[count, ind] = 1
            count += 1
        all_tensors.append(tensor_s)
    return all_tensors

def sample_sub_string(str_tensor, b):

    length = str_tensor.shape[0]
    max_len = max(length, b)
    rand1 = np.random.randint(0, length - max_len + 1)
    return str_tensor[rand1:max_len,:]


# x contains two substrings of the same review
# xk contains k negative samples all from different reviews
def sample_contrastive(all_tensors, b, k, lang):
    nwords = lang.n_words
    num_strs = len(all_tensors)
    x_ind = np.random.randint(0, num_strs)
    x = torch.zeros((b,2, n_words))
    x[:,0,:] = sample_sub_string(all_tensors[x_ind])
    x[:,1,:] = sample_sub_string(all_tensors[x_ind])
    x_k = torch.zeros((b,k,n_words))
    for i in range(k):
        x_ind = np.random.randint(0, num_strs)
        x_k[:,i,:] = sample_sub_string()
    return x, x_k

data = convert_to_one_hot(strs, lang)



class LANG_RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers = num_layers, nonlinearity = 'relu')
        return

    def forward(self, x):
        return self.rnn(x)

input_dim = lang.n_words
hidden_dim = 300
num_layers = 2
lr = 1e-3
num_iter = 1000
b = 5
k = 5
model = LANG_RNN(input_dim, hidden_dim, num_layers)
optimizer = optim.Adam(model.parameters(), lr = lr)

for t in range(num_iter):

    x, x_k = sample_contrastive(data, b, k, lang)
    fx = model.forward(x)
    fx_k = model.forward(x_k)
    fx0 = fx[:,0,:]
    fx1 = fx[:,1,:]
    loss = 0
    for i in range(k):
        fx_ki = fx_k[:,i,:]
        loss += torch.log( 1+ torch.exp(torch.dot(fx0, fx_ki) - torch.dot(fx0, fx1)) )

    print(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
