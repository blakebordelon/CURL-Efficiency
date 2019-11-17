""" Basic CNN network for testing CURL on the SVHM dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np


###
# Where to put/find SVHN dataset
svhn_path = '~/Downloads/Datasets/SVHN'
# Download if not found?
dload_dataset = False

# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_random(trainloader):
    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print labels
    print(' '.join(f'{labels[j]}' for j in range(labels.shape[0])))
    # show images
    imshow(torchvision.utils.make_grid(images))



if __name__ == '__main__':

    normalize = transforms.Compose(
                                   [transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]
                                   )
    trainset = datasets.SVHN(svhn_path, split='train', transform=normalize, target_transform=None, download=dload_dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    show_random(trainloader)


    

