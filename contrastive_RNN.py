import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import language


FORMATTED_DATA_PATH = '../formatted_data/formatted_data.txt'

def collect_formatted_strings():
    all_strings = []
    str_count = 0
    for line in open('../formatted_data/formatted_data.txt'):
        all_strings.append(line.strip())

    return all_strings



strs = collect_formatted_strings()

lang = language.Lang()
for s in strs:
    lang.addSentence(s)

def convert_to_one_hot(strs, lang):
    all_tensors = []
    str_count = 0
    for s in strs:
        sentence_length = len(s.split(' '))
        tensor_s  = torch.zeros((sentence_length, lang.n_words))
        count = 0
        for w in s.split(' '):
            ind = lang.word2index[w]
            tensor_s[count, ind] = 1
            count += 1
        all_tensors.append(tensor_s)
        str_count += 1
        
        print("str count: %d" % str_count)
    return all_tensors

def sample_sub_string(str_tensor, b):

    length = str_tensor.shape[0]
    max_len = max(length, b)
    min_len = min(length, b)
    rand1 = np.random.randint(0, length - b + 1)
    return str_tensor[rand1:rand1+b,:]


# x contains two substrings of the same review
# xk contains k negative samples all from different reviews
def sample_contrastive(all_tensors, b, k, lang):
    n_words = lang.n_words
    num_strs = len(all_tensors)
    done = False
    while done == False:
        x_ind = np.random.randint(0, num_strs)
        t = all_tensors[x_ind]
        if t.shape[0] > b:
            done = True

    x = torch.zeros((b,2, n_words))
    x[:,0,:] = sample_sub_string(all_tensors[x_ind], b)
    x[:,1,:] = sample_sub_string(all_tensors[x_ind], b)
    x_k = torch.zeros((b,k,n_words))
    for i in range(k):
        done = False
        while done == False:
            x_ind = np.random.randint(0, num_strs)
            t = all_tensors[x_ind]
            if t.shape[0] > b:
                done = True
        #x_ind = np.random.randint(0, num_strs)
        x_k[:,i,:] = sample_sub_string(all_tensors[x_ind], b)
    return x, x_k

print("converting strings to one hot tensors")
data = convert_to_one_hot(strs, lang)



class LANG_RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LANG_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers = num_layers, nonlinearity = 'relu')
        self.GRU = nn.GRU(input_dim, hidden_dim, num_layers = num_layers)
        return

    def forward(self, x):
        return self.GRU(x)
        #return self.rnn(x)

input_dim = lang.n_words
hidden_dim = 300
num_layers = 3
lr = 1e-3
num_iter = 20000
b = 10
k = 5
model = LANG_RNN(input_dim, hidden_dim, num_layers)
optimizer = optim.Adam(model.parameters(), lr = lr)

print("starting SGD")
losses = []
for t in range(num_iter):

    x, x_k = sample_contrastive(data, b, k, lang)
    fx, h = model.forward(x)
    fx_k, h = model.forward(x_k)
    fx0 = fx[:,0,:]
    fx1 = fx[:,1,:]
    loss = 0
    for i in range(k):
        fx_ki = fx_k[:,i,:]
        for j in range(b):
            loss += torch.log( 1+ torch.exp(torch.dot(fx0[j,:], fx_ki[j,:]) - torch.dot(fx0[j,:], fx1[j,:])) )

    print(loss)
    print(loss.requires_grad)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    #losses.append(loss.detach.numpy()[0])

PATH = 'IMBD_model.pt'
torch.save(model.state_dict(), PATH)

plt.plot(losses)
plt.show()
