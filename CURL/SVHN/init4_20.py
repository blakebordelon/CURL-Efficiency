""" Template for SVHN CURL training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import numpy as np
import PIL
from PIL import Image

from .. import curl

class Net_Bulk(nn.Module):
    def __init__(self):
        super(Net_Bulk, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5,80)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        return x

class Net_Head(nn.Module):
    def __init__(self):
        super(Net_Head, self).__init__()
        self.fc2 = nn.Linear(80,40)
        self.fc3 = nn.Linear(40,10)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.logsoftmax(x)
        return x

def logisticloss(D):
    """ k-way logistic loss
    """
    return torch.log2(1 + (torch.exp(D)).squeeze(-1).sum(-1))

mysoftplus=nn.Softplus()
def softplus(D):
    """ scaled softplus
    """
    return mysoftplus(D/80)

svhn_path = '~/Downloads/Datasets/SVHN'

if __name__ == '__main__':

    run = '4'
    labeledfrac = '0.020'
    unlabeledfrac = '0.080'

    normalize = transforms.Compose(
                                   [transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]
                                   )
    augcolor = [transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)]
    augaffine = [transforms.RandomAffine(20, scale=(0.9,1.1),shear=20, 
                                             resample=PIL.Image.BICUBIC, fillcolor=(100,100,100))]
    augtrans = transforms.Compose(
                [
                 transforms.RandomApply(augcolor, p=0.8),
                 transforms.RandomApply(augaffine, p=0.8),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                 ])

    head = Net_Head()
    bulk = Net_Bulk()

    curltrainer = curl.CURL(datasets.SVHN,
                            10,
                            svhn_path,
                            bulk,
                            head,
                            curlloss=softplus,
                            labeledfrac=float(labeledfrac),
                            unlabeledfrac=float(unlabeledfrac),
                            shuffle=True,
                            k=1,
                            transform=augtrans,
                            augment_transform=None,
                            augment=False,
                            testsplit='test',
                            suptrainsplit='train',
                            curltrainsplit='train',
                            groundtruth=False,
                            labeled_indices=None,
                            unlabeled_indices=None,
                            combine=True,
                            use_cuda=True,
                            download_dataset=False)

    _, sup_acc = curltrainer.train(epochs=300, batch_size=5,  test_freq=1000)
    curltrainer.get_approximate_labels()
    _, sims, conts = curltrainer.curltrain(epochs=200, batch_size=5)
    _, postcurl_acc = curltrainer.suptrain(epochs=400, batch_size=5,  test_freq=1000)

    supfile = f"CURL/SVHN/plots/sup-{run}-{labeledfrac}-{unlabeledfrac}.npy"
    unsupfile = f"CURL/SVHN/plots/unsup-{run}-{labeledfrac}-{unlabeledfrac}.npy"
    simfile = f"CURL/SVHN/plots/sim-{run}-{labeledfrac}-{unlabeledfrac}.npy"
    contfile = f"CURL/SVHN/plots/cont-{run}-{labeledfrac}-{unlabeledfrac}.npy"
    sup_arr = np.array(sup_acc)
    np.save(supfile, sup_arr)
    unsup_arr = np.array(postcurl_acc)
    np.save(unsupfile, unsup_arr)

    sims_arr = np.array(sims)
    np.save(simfile, sims_arr)
    conts_arr = np.array(conts)
    np.save(contfile, conts_arr)
