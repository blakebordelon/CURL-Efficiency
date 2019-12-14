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
        self.lin1 = nn.Linear(input_dim, int(0.6*input_dim))
        self.lin2 = nn.Linear(int(0.6*input_dim), out_dim)
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
        self.lin1 = nn.Linear(input_dim, int(0.6*input_dim))
        self.lin2 = nn.Linear(int(0.6*input_dim), out_dim)
        self.tanh = nn.Tanh()
        return
    def forward(self, x):
        x2 = self.tanh(self.lin1(x))
        x3 = self.lin2(x2)
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

    def __init__(self, block_size, k, lr = 1e-3, noise_size = 1e-4):
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.x_train_sup = []
        self.y_train_sup = []
        self.training_class_inds = []
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
        self.training_class_inds = []
        min = 50000
        for i in range(10):
            arr = [j for j in range(len(y_train)) if y_train[j]==i]
            if len(arr)<min:
                min = len(arr)
            self.training_class_inds.append(arr)
        true_arr = np.zeros((10,min))
        for i in range(10):
            true_arr[i,:] = self.training_class_inds[i][0:min]
        self.training_class_inds = true_arr
        return

    def set_supervised_data_set(self, x_train, y_train):
        self.x_train_sup = x_train
        self.y_train_sup = y_train
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
        min_count = self.training_class_inds.shape[1]
        inds_indsx = np.random.randint(0,min_count,self.block_size)
        inds_indsxp = np.random.randint(0, min_count, self.block_size)

        x_inds = np.zeros(self.block_size).astype('int')
        inds_xp = np.zeros(self.block_size).astype('int')
        for i in range(self.block_size):
            c_i = c[i]
            x_inds[i] = self.training_class_inds[c_i,inds_indsx[i]]
            inds_xp[i] = self.training_class_inds[c_i,inds_indsxp[i]]

        """
        sort_inds = np.argsort(self.y_train)
        x_inds = np.random.randint(0, len(self.y_train), self.block_size)
        x_class = self.y_train[x_inds]
        inds_xp = []
        for i in range(len(x_inds)):
            c = x_class[i]
            other_class_inds = [j for j in range(len(self.y_train)) if self.y_train[j] == c]
            inds_xp.append(other_class_inds[np.random.randint(len(other_class_inds))])
        """

        #xp_inds = sort_inds[x_class*int(len(self.y_train)/10) + np.random.randint(0, int(len(self.y_train)/10), self.block_size)]
        #xp_inds = sort_inds[ x_class + np.random.randint(0, int(len(self.y_train)/10), self.block_size )]
        sample_mistakes = (self.y_train[x_inds] != self.y_train[inds_xp]).sum()
        print(sample_mistakes)
        xk_inds = np.random.randint(0, len(self.y_train), (self.block_size*self.k))
        x = torch.tensor(self.x_train[x_inds,:],dtype=torch.float)
        xp = torch.tensor(self.x_train[inds_xp,:],dtype=torch.float)
        xk = torch.tensor(self.x_train[xk_inds,:],dtype=torch.float)
        return x,xp,xk

    def sample_supervised_train(self):
        num_train = len(self.y_train_sup)
        dim = self.x_train_sup.shape[1]
        inds = np.random.randint(0, num_train, self.block_size)
        x = torch.tensor(self.x_train_sup[inds,:], dtype = torch.float)
        y = torch.tensor(self.y_train_sup[inds], dtype = torch.long)
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
        optimizer = optim.Adam(curl_net.parameters(), lr = self.lr)
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
            loss = 0
            pos_prod = (fx * fxp).sum(-1)
            fxrep = fx.repeat(self.k,1,1).permute(1,2,0)
            fxmreshape = fxm.reshape(self.k, self.block_size, fx.shape[1])
            neg_prods = (fxm.reshape(self.k, self.block_size, fx.shape[1]) * fx.repeat(self.k,1, 1)).sum(2)


            # pos_prod is block-size ; neg_prods is k x block-size
            pos_prod_rep = pos_prod.repeat(self.k, 1)
            arg = torch.exp(neg_prods - pos_prod_rep).sum(0)
            loss = 1/(self.block_size*self.k) * torch.log(torch.ones_like(arg) + arg).sum()

            #loss = torch.log( torch.ones_like(F12) + torch.exp( (F13) / curl_net.out_dim )).mean().mean()
            #loss = torch.max(torch.zeros_like(F12), torch.ones_like(F12) + torch.max(F13-F12)).mean().mean()
            #print(loss.requires_grad)
            #print(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_val = loss.detach().numpy()
            losses.append(loss_val)
            print("CURL Loss: %lf" % loss_val)

        plt.figure()
        plt.plot(losses)
        plt.title('CURL Training')
        plt.ylabel('CURL Loss')
        plt.xlabel('iterations')
        plt.savefig('unsup_loss.pdf')
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

    def visualize(self, curl_net, name):
        x = torch.tensor(self.x_test,dtype = torch.float)
        x.requires_grad = False
        fx = curl_net(x)
        f_numpy = fx.detach().numpy()
        u,s,vh = np.linalg.svd(f_numpy)
        ind_sort = np.argsort(s)[::-1]
        proj = f_numpy @ vh[ind_sort[0:2],:].T

        plt.figure()
        for i in range(10):
            inds_i = [j for j in range(len(self.y_test)) if self.y_test[j]==i]
            plt.scatter(proj[inds_i,0], proj[inds_i,1])
        plt.savefig(name+'.pdf')
        return


    def test_sup(self, curl_net, sup_net):
        curl_net.eval()
        sup_net.eval()
        x = torch.tensor(self.x_test, dtype = torch.float)
        y = torch.tensor(self.y_test, dtype = torch.long)
        fx = curl_net.forward(x)
        y_hat = sup_net.forward(fx).argmax(dim =1)
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


    def train_autoencoder_CURL(self, encoder, decoder, num_iter, with_CURL = True):
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
            reconstruction_loss += (x-xhat).pow(2).mean()
            reconstruction_loss += (xphat-xp).pow(2).mean()
            reconstruction_loss += (xm-xmhat).pow(2).mean()

            loss = 0
            pos_prod = (fx * fxp).sum(-1)
            fxrep = fx.repeat(self.k,1,1).permute(1,2,0)
            fxmreshape = fxm.reshape(self.k, self.block_size, fx.shape[1])
            neg_prods = (fxm.reshape(self.k, self.block_size, fx.shape[1]) * fx.repeat(self.k,1, 1)).sum(2)


            # pos_prod is block-size ; neg_prods is k x block-size
            pos_prod_rep = pos_prod.repeat(self.k, 1)
            arg = torch.exp(neg_prods - pos_prod_rep).sum(0)
            loss = 1/(self.block_size*self.k) * torch.log(torch.ones_like(arg) + arg).sum()


            total_loss = 0.5* reconstruction_loss
            if with_CURL==True:
                total_loss += loss
            #total_loss = reconstruction_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("reconstruction_loss: %lf" % reconstruction_loss.detach().numpy())
            print("CURL loss: %lf" % loss.detach().numpy())

        return encoder, decoder


image_shape = 28
input_dim = image_shape**2
output_dim = 120
block_size = 150
k = 5
num_iter = 2000
num_classes = 10


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (x_train.shape[0], input_dim)) / 255.0
x_test = np.reshape(x_test, (x_test.shape[0], input_dim)) / 255.0
curl_net = CURL_NET(input_dim, output_dim)
sup_net = SUP_NET(output_dim, num_classes)
trainer = Train(block_size, k)

#M_vals = [int(0.1*len(y_train) ), int(0.25*len(y_train)),  int(1/2*len(y_train)), int(0.75 * len(y_train)) , len(y_train)]
M_vals = [50, 100, 500, 1000, 5000, 10000, 20000, 50000]
#N_vals = [100, 500, 1000, 5000]
N_vals = [25, 100, 500, 1000]
CURL_errors = np.zeros((len(M_vals), len(N_vals)))
AE_errors = np.zeros(( len(M_vals), len(N_vals) ))
RES_errors = np.zeros((len(M_vals), len(N_vals)))
SUP_errors = np.zeros((len(M_vals), len(N_vals) ) )


for i in range(len(M_vals)):

    m = M_vals[i]

    inds_m = np.random.randint(0, len(y_train), m)
    trainer.set_data_set(x_train[inds_m,:], x_test, y_train[inds_m], y_test)

    # CURL + AE
    encoder = Encoder(input_dim, output_dim)
    decoder = Decoder(output_dim, input_dim)
    trainer.visualize(encoder, 'autoencoder_before')
    encoder, decoder = trainer.train_autoencoder_CURL(encoder, decoder, num_iter)
    sup_net = SUP_NET(output_dim, num_classes)
    print("calling train sup for autoencoder")

    for j in range(len(N_vals)):
        n = N_vals[j]
        inds_n = np.random.randint(0, len(y_train), n)
        trainer.set_supervised_data_set(x_train[inds_n,:], y_train[inds_n])
        sup_net = trainer.train_sup(encoder, sup_net, 1000)
        test_risk_ae = trainer.test_sup(encoder, sup_net)
        trainer.visualize(encoder, 'autoencoder_after')
        print("test risk ae: %lf" % test_risk_ae)
        AE_errors[i,j] = test_risk_ae

        curl_net = CURL_NET(input_dim, output_dim)
        trainer.visualize(curl_net, 'pure_sup_before')
        sup_net = SUP_NET(output_dim, num_classes)
        sup_net, curl_net = trainer.train_pure_sup(curl_net, sup_net, 1000)
        error_sup_pure = trainer.test_sup(curl_net, sup_net)
        trainer.visualize(curl_net, 'pure_sup_after')
        SUP_errors[i,j] = error_sup_pure
        print("error pure supervised: %lf" % error_sup_pure )


    # CURL TEST
    curl_net = CURL_NET(input_dim, output_dim)
    curl_net = trainer.train_unsup(curl_net, num_iter)
    trainer.visualize(curl_net, 'curl_after')
    for j in range(len(N_vals)):
        n = N_vals[j]
        inds_n = np.random.randint(0, len(y_train), n)
        trainer.set_supervised_data_set(x_train[inds_n,:], y_train[inds_n])
        sup_net = trainer.train_sup(curl_net, sup_net, 1000)
        test_risk_curl = trainer.test_sup(curl_net, sup_net)
        CURL_errors[i,j] = test_risk_curl



plt.figure()
for i in range(len(N_vals)):
    plt.semilogx(M_vals, CURL_errors[:,i], label = 'N = %d' % N_vals[i])
plt.legend()
plt.xlabel('M')
plt.ylabel('Test Risk')
plt.savefig('ground_truth_curl_pretraining_d%d.pdf' % output_dim)


plt.figure()
for i in range(len(N_vals)):
    plt.semilogx(M_vals, AE_errors[:,i], label = 'N = %d' % N_vals[i])
plt.legend()
plt.xlabel('M')
plt.ylabel('Test Risk')
plt.savefig('ground_truth_ae_pretraining_d%d.pdf' %output_dim)


plt.figure()
for i in range(len(N_vals)):
    plt.semilogx(M_vals, SUP_errors[:,i] - CURL_errors[:,i], label = 'N = %d' % N_vals[i])
plt.legend()
plt.xlabel('M')
plt.ylabel('Acc Boost From CURL')
plt.savefig('curl_boost_d_%d.pdf' % output_dim)


plt.figure()
for i in range(len(N_vals)):
    plt.semilogx(M_vals, SUP_errors[:,i] - AE_errors[:,i], label = 'N = %d' % N_vals[i])
plt.legend()
plt.xlabel('M')
plt.ylabel('Acc Boost From CURL+AE')
plt.savefig('ae_boost_d_%d.pdf' % output_dim)
