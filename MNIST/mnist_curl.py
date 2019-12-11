import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow.keras as keras
from torch import autograd
import matplotlib.pyplot as plt
# experiment with simple perturbed positive samples
class CURL_NET(nn.Module):

    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        #self.conv1 = nn.Conv2d(input_dim, 6, 50)
        self.lin1 = nn.Linear(input_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()
        return

    def forward(self, x):

        x2 = self.tanh(self.lin1(x))
        x3 = self.tanh(self.lin2(x2))
        return x3

class SUP_NET(nn.Module):

    def __init__(self, hidden_dim, readout_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_dim = readout_dim
        self.lin_read = nn.Linear(hidden_dim, readout_dim)
        self.sm = nn.LogSoftmax()

    def forward(self, x):
        return self.sm(self.lin_read(x))

class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        #self.conv1 = nn.Conv2d(input_dim, 6, 50)
        self.lin1 = nn.Linear(input_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()
        return
    def forward(self, x):
        x2 = self.tanh(self.lin1(x))
        x3 = self.tanh(self.lin2(x2))
        return x3

class Decoder(nn.Module):

    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        #self.conv1 = nn.Conv2d(input_dim, 6, 50)
        self.lin1 = nn.Linear(input_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()
        return
    def forward(self, x):
        x2 = self.tanh(self.lin1(x))
        x3 = self.lin2(x2)
        return x3


class Train:

    def __init__(self, block_size, k, lr = 2e-3, noise_size = 1e-4):
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.block_size = block_size
        self.k = k
        self.lr = lr
        self.noise_size = noise_size

        return

    def set_data_set(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        return

    def sample_contrastive_noise(self):

        num_train = len(self.y_train)
        dim = self.x_train.shape[1]
        indices1 = np.random.randint(0, num_train, self.block_size)
        indices2 = np.random.randint(0, num_train, self.block_size* self.k)
        x = self.x_train[indices1,:]
        xp = x + np.random.multivariate_normal(np.zeros(x.shape[1]), self.noise_size*np.eye(x.shape[1]), self.block_size)
        xm = self.x_train[indices2,:]
        x = torch.tensor(x, dtype = torch.float)
        xp = torch.tensor(xp, dtype = torch.float)
        xm = torch.tensor(xm, dtype = torch.float)
        x.requires_grad = True
        xp.requires_grad = True
        xm.requires_grad = True

        return x, xp, xm

    def sample_contrastive_correct(self):
        c = np.random.randint(0,10, self.block_size)
        sort_inds = np.argsort(self.y_train)
        x_inds = np.random.randint(0, len(self.y_train), self.block_size)
        x_class = self.y_train[x_inds]
        inds_xp = []
        for i in range(len(x_inds)):
            c = x_class[i]
            other_class_inds = [j for j in range(len(self.y_train)) if self.y_train[j] == c]
            inds_xp.append(other_class_inds[np.random.randint(len(other_class_inds))])

        #xp_inds = sort_inds[x_class*int(len(self.y_train)/10) + np.random.randint(0, int(len(self.y_train)/10), self.block_size)]
        #xp_inds = sort_inds[ x_class + np.random.randint(0, int(len(self.y_train)/10), self.block_size )]
        sample_mistakes = (self.y_train[x_inds] != self.y_train[inds_xp]).sum()
        xk_inds = np.random.randint(0, len(self.y_train), self.block_size*self.k)
        x = torch.tensor(self.x_train[x_inds,:],dtype=torch.float)
        xp = torch.tensor(self.x_train[inds_xp,:],dtype=torch.float)
        xk = torch.tensor(self.x_train[xk_inds,:],dtype=torch.float)
        return x,xp,xk

    def sample_supervised_train(self):
        num_train = len(self.y_train)
        dim = self.x_train.shape[1]
        inds = np.random.randint(0, num_train, self.block_size)
        x = torch.tensor(self.x_train[inds,:], dtype = torch.float)
        y = torch.tensor(self.y_train[inds], dtype = torch.long)
        return x,y

    def sample_supervised_test(self):
        num_test = len(self.y_test)
        dim = self.x_test.shape[1]
        inds = np.random.randint(0, num_test, self.block_size)
        x = torch.tensor(self.x_test[inds,:], dtype = torch.float)
        y = torch.tensor(self.y_test[inds], dtype = torch.long)
        return x,y

    def train_unsup(self, curl_net, num_iter):

        curl_net.train()
        optimizer = optim.Adam(curl_net.parameters(), lr =1e-3)
        avg_loss = 0
        losses = []
        for t in range(num_iter):
            #x, xp, xm = self.sample_contrastive_noise()
            x,xp,xm = self.sample_contrastive_correct()
            #print(x.requires_grad)
            fx = curl_net.forward(x)
            fxp = curl_net.forward(xp)
            fxm = curl_net.forward(xm)
            #print(fx.requires_grad)
            F1 = fx.reshape((fx.shape[0], 1, fx.shape[1]))
            F2 = fxp.reshape((fxp.shape[0], fxp.shape[1], 1))
            #print(F1.requires_grad)
            F3 = fxm.reshape((self.block_size, fxm.shape[1], self.k))
            F12 = torch.bmm(F1,F2).squeeze().unsqueeze(-1).repeat(1,self.k) # should be a vector of batchsize
            F13 = torch.bmm(F1,F3).squeeze() # should be a batchsize x k matrix
            #loss = torch.log( torch.ones_like(F12) + torch.exp( 0.1 *(F13 - F12 + torch.ones_like(F13)) / curl_net.out_dim )).mean().mean()
            loss = torch.log( torch.ones_like(F12) + torch.exp( (F13) / curl_net.out_dim )).mean().mean()
            #loss = torch.max(torch.zeros_like(F12), torch.ones_like(F12) + torch.max(F13-F12)).mean().mean()
            #print(loss.requires_grad)
            #print(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_val = loss.detach().numpy()
            losses.append(loss_val)
        plt.plot(losses)
        plt.title('CURL Training')
        plt.ylabel('CURL Loss')
        plt.xlabel('iterations')
        plt.savefig('unsup_loss.pdf')
        plt.show()
        return curl_net

    def train_sup(self, curl_net, sup_net, num_iter):

        curl_net.eval()
        sup_net.train()
        optimizer = optim.Adam(sup_net.parameters(), lr=self.lr)
        criterion = nn.NLLLoss()
        for t in range(num_iter):
            x,y = self.sample_supervised_train()
            y_hat = sup_net.forward(curl_net.forward(x))
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        x = torch.tensor(self.x_train, dtype = torch.float)
        y = torch.tensor(self.y_train, dtype = torch.long)
        y_hat = sup_net.forward(curl_net.forward(x)).argmax(dim =1)
        error = (y_hat != y).sum().item() / y.shape[0]
        print("train error")
        print(error)
        return sup_net

    def test_sup(self, curl_net, sup_net):
        curl_net.eval()
        sup_net.eval()
        x = torch.tensor(self.x_test, dtype = torch.float)
        y = torch.tensor(self.y_test, dtype = torch.long)
        y_hat = sup_net.forward(curl_net.forward(x)).argmax(dim =1)
        error = (y_hat != y).sum().item() / y.shape[0]
        print("test error")
        print(error)
        return error

    def train_pure_sup(self, curl_net, sup_net, num_iter):
        curl_net.train()
        sup_net.train()
        optimizer = optim.Adam(list(curl_net.parameters()) + list(sup_net.parameters()), lr=self.lr)
        criterion = nn.NLLLoss()
        for t in range(num_iter):
            x,y = self.sample_supervised_train()
            y_hat = sup_net.forward(curl_net.forward(x))
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        x = torch.tensor(self.x_train, dtype = torch.float)
        y = torch.tensor(self.y_train, dtype = torch.long)
        y_hat = sup_net.forward(curl_net.forward(x)).argmax(dim =1)
        error = (y_hat != y).sum().item() / y.shape[0]
        print("train error")
        print(error)
        return sup_net, curl_net


    def train_autoencoder_CURL(self, encoder, decoder, num_iter):
        encoder.train()
        decoder.train()
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = self.lr)
        for t in range(num_iter):
            x,xp,xm = self.sample_contrastive_correct()
            fx = encoder.forward(x)
            fxp = encoder.forward(xp)
            fxm = encoder.forward(xm)

            xhat = decoder.forward(fx)
            xphat = decoder.forward(fxp)
            xmhat = decoder.forward(fxm)

            reconstruction_loss = 0
            reconstruction_loss += (x-xhat).pow(2).mean().mean()
            reconstruction_loss += (xphat-xp).pow(2).mean().mean()
            reconstruction_loss += (xm-xmhat).pow(2).mean().mean()

            F1 = fx.reshape((fx.shape[0], 1, fx.shape[1]))
            F2 = fxp.reshape((fxp.shape[0], fxp.shape[1], 1))
            F3 = fxm.reshape((self.block_size, fxm.shape[1], self.k))
            F12 = torch.bmm(F1,F2).squeeze().unsqueeze(-1).repeat(1,self.k) # should be a vector of batchsize
            F13 = torch.bmm(F1,F3).squeeze() # should be a batchsize x k matrix
            loss = torch.log( torch.ones_like(F12) + torch.exp( (F13 - F12 + torch.ones_like(F12)) / curl_net.out_dim )).mean().mean()

            #total_loss = loss + reconstruction_loss
            total_loss = reconstruction_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("reconstruction_loss: %lf" % reconstruction_loss.detach().numpy())
            print("CURL loss: %lf" % loss.detach().numpy())

        return encoder, decoder


image_shape = 28
input_dim = image_shape**2
output_dim = 250
block_size = 25
k = 5
num_iter = 120
num_classes = 10


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (x_train.shape[0], input_dim)) / 255.0
x_test = np.reshape(x_test, (x_test.shape[0], input_dim)) / 255.0
curl_net = CURL_NET(input_dim, output_dim)
sup_net = SUP_NET(output_dim, num_classes)
trainer = Train(block_size, k)

M_vals = [len(y_train)]
N_vals = [100, 500, 1000, 5000]
errors = np.zeros((len(M_vals), len(N_vals)))
for i in range(len(M_vals)):
    for j in range(len(N_vals)):
        m = M_vals[i]
        n = N_vals[j]
        inds_m = np.random.randint(0, len(y_train), m)
        inds_n = np.random.randint(0, len(y_train), n)

        trainer.set_data_set(x_train[inds_m,:], x_test, y_train[inds_m], y_test)
        encoder = Encoder(input_dim, output_dim)
        decoder = Decoder(output_dim, input_dim)
        encoder, decoder = trainer.train_autoencoder_CURL(encoder, decoder, num_iter)
        sup_net = SUP_NET(output_dim, num_classes)
        sup_net = trainer.train_sup(encoder, sup_net, 1000)



        curl_net = CURL_NET(input_dim, output_dim)
        sup_net = SUP_NET(output_dim, num_classes)
        trainer.set_data_set(x_train[inds_m,:], x_test, y_train[inds_m], y_test)
        curl_net = trainer.train_unsup(curl_net, num_iter)
        trainer.set_data_set(x_train[inds_n,:], x_test, y_train[inds_n], y_test)
        sup_net = trainer.train_sup(curl_net, sup_net, 1000)
        #sup_net, curl_net = trainer.train_pure_sup(curl_net, sup_net, 1000)
        errors[i,j] = trainer.test_sup(curl_net, sup_net)

        curl_net = CURL_NET(input_dim, output_dim)
        sup_net = SUP_NET(output_dim, num_classes)
        sup_net = trainer.train_sup(curl_net, sup_net, 1000)
        error_sup_resevoir = trainer.test_sup(curl_net, sup_net)
        print("error resevoir-supervised: %lf" % error_sup_resevoir )

        curl_net = CURL_NET(input_dim, output_dim)
        sup_net = SUP_NET(output_dim, num_classes)
        sup_net, curl_net = trainer.train_pure_sup(curl_net, sup_net, 1000)
        error_sup_pure = trainer.test_sup(curl_net, sup_net)
        print("error pure supervised: %lf" % error_sup_pure )

for i in range(len(N_vals)):
    plt.plot(M_vals, errors[:,i], label = 'N = %d' % N_vals[i])
plt.title('no CURL Pretraining')
plt.legend()
plt.xlabel('M')
plt.ylabel('Test Risk')
plt.savefig('ground_truth_curl_pretraining.pdf')
plt.show()

curl_net = CURL_NET(input_dim, output_dim)
sup_net = SUP_NET(output_dim, num_classes)
sup_net, curl_net = trainer.train_pure_sup(curl_net, sup_net, 1000)
trainer.test_sup(curl_net, sup_net)
