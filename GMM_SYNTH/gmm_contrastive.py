import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def sample_GMM(num_centers, num_samples, dim, var):

    #sigma = np.array([[1,0.8],[0.8,1]])
    A = np.random.normal(0,1,(dim, dim))
    sigma = A.T @ A
    centers = np.random.multivariate_normal(np.zeros(dim), sigma, num_centers)
    X = np.zeros((num_centers*num_samples, dim))
    for i in range(num_centers):
        X[i*(num_samples):(i+1)*num_samples,:] = np.random.multivariate_normal(centers[i,:], var*np.eye(dim), num_samples)
    return X, centers

def plot_clusters(X, num_centers, num_samples, file_name = 'cluster_plot'):
    for i in range(num_centers):
        plt.scatter(X[i*(num_samples):(i+1)*num_samples,0], X[i*(num_samples):(i+1)*num_samples,1])
    plt.savefig(file_name+'.pdf')
    plt.show()


def corrupt(X, up_dim, nonlinearity='tanh'):
    up_project = 1/dim * np.random.normal(0,1, (dim, up_dim))
    #A = np.random.normal(0,1, (dim, dim))
    X1 = X @ up_project
    if nonlinearity=='tanh':
        X_new = np.tanh(X1)
    elif nonlinearity =='relu':
        X_new = np.maximum(X1, np.zeros(X1.shape))
    elif nonlinearity == 'cosine':
        X_new = np.cos(X1)
    down_project = np.random.normal(0,1, (up_dim, dim))
    X_new = 1/up_dim * X_new @ down_project
    if nonlinearity == 'relu':
        X_new = np.maximum(X_new, np.zeros(X_new.shape))
    plot_clusters(X_new, num_centers, num_samples)
    return X_new

def analytic_solve(centers):
    num_centers, dim = centers.shape
    Sigma = np.cov(centers.T)
    u, s, v = np.linalg.svd(Sigma)
    sqrt = np.diag(np.power(s,-1/2))
    return sp.linalg.sqrtm(np.linalg.inv(Sigma))
    #return np.linalg.inv(Sigma)

def contrastive_optimization(X, num_centers, num_samples, num_iter, k=5):
    dim = X.shape[1]
    A = np.eye(dim)
    eta = 5e-3
    losses = []
    num_averages = 200
    iter_avg = int(num_iter / num_averages)
    avg_loss = 0

    for t in range(num_iter):
        c1= np.random.randint(0, num_centers)
        c2 = np.random.randint(0, num_centers)
        while c2 == c1:
            c2 = np.random.randint(0, num_centers)

        rand_x = np.random.randint(0,num_samples)
        rand_xp = np.random.randint(0, num_samples)
        for i in range(k):
            rand_xm = np.random.randint(0, num_samples)

            x = X[c1*num_samples + rand_x,:]
            xp = X[c1*num_samples + rand_xp,:]
            xm = X[c2*num_samples + rand_xm,:]
            grad = np.zeros((dim,dim))
            val = np.dot(x, A.T @ A @ (xm-xp))
            print("contrastive loss")
            loss = np.maximum(val, 0)

            avg_loss += 1/iter_avg * loss
            if t % iter_avg == 0 and t>0:
                losses.append(avg_loss)
                avg_loss = 0

            if val > 0:
                grad = A @ (np.outer(x, xm-xp) + np.outer(xm-xp,x))
            A += - eta* grad
            print(np.linalg.norm(A,'fro'))

    plt.plot(losses)
    plt.xlabel('epochs (50 samples)')
    plt.ylabel('contrastive loss Linear model')
    plt.savefig('contrastive_loss_training.pdf')
    plt.show()
    return A

def random_projection_analytic(centers, up_dim):
    dim = centers.shape[1]
    W_up = np.random.random_sample((dim, up_dim))
    X_pre = centers @ W_up
    #phi = np.maximum(X_pre, np.zeros(X_pre.shape))
    phi = np.tanh(X_pre)
    Sigma_phi = np.cov(phi.T)
    u, s, v = np.linalg.svd(Sigma_phi)
    A = u[:,0:2].T
    sqrt_Lambda = np.diag(np.power(s,-1/2))
    W_down = A @ sqrt_Lambda @ u.T
    print(sqrt_Lambda @ u.T @ Sigma_phi @ v @ sqrt_Lambda)
    eye_hopeful = W_down @ Sigma_phi @ W_down.T
    print("eye hopeful: ")
    print(eye_hopeful)
    W_down = W_down.T

    print(W_down.shape)
    return W_up, W_down


class CURL_NET(nn.Module):

    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.lin1 = nn.Linear(input_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()
        return

    def forward(self, x):

        x2 = self.tanh(self.lin1(x))
        x3 = self.tanh(self.lin2(x2))
        return x3

class Sup_Net(nn.Module):

    def __init__(self, hidden_dim, readout_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_dim = readout_dim
        self.lin_read = nn.Linear(hidden_dim, readout_dim)
        self.sm = nn.LogSoftmax()

    def forward(self, x):
        return self.sm(self.lin_read(x))

class TotalNet(nn.Module):

    def __init__(self, curl_net, sup_net):
        super().__init__()
        self.curl_net = curl_net
        self.sup_net = sup_net

    def forward(self, x):
        out = self.curl_net.forward(x)
        return self.sup_net.forward(out)

class DataSet:
    def __init__(self, num_centers, num_samples, dim, var):
        self.num_centers = num_centers
        self.num_samples = num_samples
        self.dim = dim
        self.var = var
        X, centers = sample_GMM(num_centers, num_samples, dim, var)
        self.X = X
        self.centers = centers

    def sample_contrastive(self, k, collision = False):

        c1 = np.random.randint(0, self.num_centers)
        c2 = np.random.randint(0, self.num_centers)
        if collision == False:
            while c2 == c1:
                c2 =np.random.randint(0, self.num_centers)

        ind_x  = c1*num_samples + np.random.randint(0, self.num_samples)
        ind_xp = c1*num_samples + np.random.randint(0, self.num_samples)
        ind_xk = (c2*num_samples*np.ones(k) + np.random.randint(0, self.num_samples, k)).astype('int')
        x = torch.zeros((1, self.dim))
        xp = torch.zeros((1,self.dim))
        xk = torch.zeros((k, self.dim))
        x[0,:] = torch.tensor(self.X[ind_x,:])
        xp[0,:] = torch.tensor(self.X[ind_xp,:])
        xk[:,:] = torch.tensor(self.X[ind_xk,:])
        return x, xp, xk

    def sample_supervised(self, batch_size):
        randint = np.random.randint(0, self.num_samples*self.num_centers, batch_size)
        y_ints = torch.tensor(randint/self.num_samples, dtype = torch.long)
        x = torch.tensor(self.X[randint], dtype = torch.float)
        return x, y_ints



class Train:

    def __init__(self, num_iter, k, data_set, lr = 1e-4):
        self.num_iter = num_iter
        self.k = k
        self.lr = lr
        self.data_set = data_set
        return

    def train_CURL(self, model, up_dim, dim):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr = self.lr)
        losses_un = []
        losses_sup = []
        plot_clusters(self.data_set.X, self.data_set.num_centers, self.data_set.num_samples, file_name = 'before')

        avg_loss = 0
        plot_every = 250
        for t in range(self.num_iter):
            x, xp, xk = self.data_set.sample_contrastive(self.k)
            fx = model.forward(x).squeeze()
            fxp = model.forward(xp).squeeze()
            fxk = model.forward(xk).squeeze()
            loss = 0
            for i in range(k):
                loss += torch.log(1 + torch.exp(torch.dot(fx, fxk[i,:]) - torch.dot(fx,fxp)))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_t = loss.detach().numpy()
            avg_loss += loss_t / plot_every
            if t % plot_every == 0 and t>0:
                losses_un.append(avg_loss)
                avg_loss = 0
                sup_net = Sup_Net(curl_net.out_dim, self.data_set.num_centers)
                sup_net, error = self.train_sup(curl_net, sup_net, 500)
                curl_net.train()
                losses_sup.append(error)

        plt.plot(losses_un)
        plt.xlabel('CURL Epochs')
        plt.ylabel('CURL Loss')
        plt.savefig('unsup_loss.pdf')
        plt.show()
        plt.plot(losses_sup)
        plt.xlabel('CURL Epochs')
        plt.ylabel('Supervised Loss')
        plt.savefig('sup_loss.pdf')
        plt.show()
        model.eval()
        X_new = model.forward(torch.tensor(self.data_set.X, dtype = torch.float)).detach().numpy()
        plot_clusters(X_new, self.data_set.num_centers, self.data_set.num_samples, file_name = 'after')
        return model

    def train_sup(self, curl_net, sup_net, num_iter):

        curl_net.eval()
        sup_net.train()
        criterion = nn.NLLLoss()
        optimizer=optim.Adam(sup_net.parameters(), lr=1e-3)
        batch_size = 100
        for t in range(num_iter):
            x, y = self.data_set.sample_supervised(batch_size)
            #x = torch.tensor(x)
            #y = torch.tensor(y, dtype = torch.long)
            fx = curl_net.forward(x)
            y_hat = sup_net.forward(fx)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_val = loss.detach().numpy()
            #print("sup loss = %lf" % loss_val)

        sup_net.eval()
        num_eval = 100
        error  = 0
        for t in range(num_eval):
            x,y = self.data_set.sample_supervised(batch_size)
            y_hat = sup_net.forward(curl_net.forward(x)).argmax(dim=1)
            error += 1/num_eval *  (y_hat != y).sum().item() / batch_size

        print(error)
        return sup_net, error




num_centers = 30
num_samples = 50
dim = 2
up_dim = 20
var = 0.005
num_iter = 200000
k = 10

data_set = DataSet(num_centers, num_samples, dim, var)
curl_net = CURL_NET(dim, up_dim)
sup_net = Sup_Net(up_dim, dim)
total_net = TotalNet(curl_net, num_centers)
trainer = Train(num_iter, k, data_set)
trainer.train_CURL(curl_net, up_dim, dim)




W_up, W_down = random_projection_analytic(centers, up_dim)
X_new = np.maximum(X @ W_up, np.zeros((X.shape[0], W_up.shape[1]))) @ W_down
plot_clusters(X_new, num_centers, num_samples, 'random_proj_analytic')


plot_clusters(X, num_centers, num_samples, 'before')
A_optimal = analytic_solve(centers)
X_opt = X @ A_optimal
plot_clusters(X_opt, num_centers, num_samples, 'after')
A_num = contrastive_optimization(X, num_centers, num_samples, num_iter)
X_num = X @ A_num
plot_clusters(X_num, num_centers, num_samples, 'numerical')

#up_dim = 10
#X_new = corrupt(X, up_dim, nonlinearity = 'tanh')
