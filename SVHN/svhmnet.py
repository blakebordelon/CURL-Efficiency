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

class Net_Bulk(nn.Module):
    def __init__(self):
        super(Net_Bulk, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5,120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # TODO check what these dimensions mean
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        return x

class Net_Head(nn.Module):
    def __init__(self):
        super(Net_Head, self).__init__()
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.logsoftmax(x)
        return x

class Net_Full(nn.Module):
    def __init__(self):
        super(Net_Full, self).__init__()
        self.bulk = Net_Bulk()
        self.head = Net_Head()

    def forward(self, x):
        x = self.bulk(x)
        x = self.head(x)
        return x


class Train_Sup():
    def __init__(self, trainloader, use_cuda=False):
        self.net = Net_Full()
        self.trainloader = trainloader
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        self.net.to(self.device)

    def train(self, epochs=10, loss_freq=1):
        self.net.train()
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        train_losses = []
        for epoch in range(epochs):
            t = tqdm(self.trainloader, leave=False, total=len(self.trainloader))
            for i, data in enumerate(t):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # record loss
                loss_val = loss.cpu().item()
                if i % loss_freq == 0:
                    train_losses.append(loss_val)
                t.set_postfix(epoch=f'{epoch}/{epochs-1}', loss=f'{loss_val:.2e}')
        self.net.eval()
        return train_losses

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
    testset = datasets.SVHN(svhn_path, split='test', transform=normalize, target_transform=None, download=dload_dataset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

    trainsup = Train_Sup(trainloader, use_cuda=True)
    for epoch in range(10):
        trainsup.train(epochs=1)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(trainsup.device), data[1].to(trainsup.device)
                outputs = trainsup.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch}: Accuracy of the network on the test images: {100 * correct/total}')


