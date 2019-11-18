""" Basic CNN network for testing CURL on the SVHM dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import PIL
from PIL import Image


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
    def __init__(self, svhn_path, frac=0.5, shuffle=True, augment=True, use_cuda=False):
        """
        frac : float
            fraction of dataset to use for training
        """

        self.net = Net_Full()
        normalize = transforms.Compose(
                     [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ])
        augcolor = [transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)]
        augaffine = [transforms.RandomAffine(20, scale=(0.9,1.1),shear=20, 
                                                 resample=PIL.Image.BICUBIC, fillcolor=(100,100,100))]
        augtrans = transforms.Compose(
                    [
                     transforms.RandomApply(augcolor, p=0.8),
                     transforms.RandomApply(augcolor, p=0.8),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ])
        if augment:
            transform = normalize
        else:
            transform = augtrans
        trainset = datasets.SVHN(svhn_path, split='train', transform=transform, target_transform=None, download=dload_dataset)
        self.trainset = trainset

        testset = datasets.SVHN(svhn_path, split='test', transform=normalize, target_transform=None, download=dload_dataset)
        self.testset = testset

        trainset_size = len(self.trainset)
        indices = list(range(trainset_size))
        end = int(np.floor(frac * trainset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices = indices[:end]
        self.train_sampler = SubsetRandomSampler(train_indices)

        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("CUDA not available")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        self.net.to(self.device)

    def train(self, batch_size=8, epochs=10, loss_freq=1, test_freq=10000):
        self.net.train()
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, sampler=self.train_sampler, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        train_losses = []
        bnum = 0
        #t = tqdm(leave=True, total=len(epochs*len(trainloader)))
        t = tqdm(leave=True, total=epochs*len(trainloader))
        for epoch in range(epochs):
            for i, data in enumerate(trainloader):

                if bnum % test_freq == 0:
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data[0].to(trainsup.device), data[1].to(trainsup.device)
                            outputs = trainsup.net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    print(f'Epoch {epoch}: test accuracy = {100 * correct/total}')

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
                bnum += 1
                t.update()
                t.set_postfix(epoch=f'{epoch}', loss=f'{loss_val:.2e}')
        t.close()
        self.net.eval()
        return train_losses


class ContrastedData(datasets.SVHN):
    def __init__(self, root, split='train', contrast_transform=None, k=1, 
                 transform=None, target_transform=None, download=False):
        super(ContrastedData, self).__init__(root, split=split, transform=transform,  
                                             target_transform=target_transform, download=download)
        self.contrast_transform = contrast_transform
        self.k = k
        if transform is None:
            print("Warning - a transform must be provided, at the very least transform.ToTensor()")
            print("After transformation, the result should be a tensor.")

    def __getitem__(self, index):
        imgs = []
        targets = []  # We actually shouldn't be using the target for CURL anyways
        # Create original
        img_base, target_base = self.data[index], int(self.labels[index])
        imgx = Image.fromarray(np.transpose(img_base, (1, 2, 0)))
        if self.transform is not None:
            imgx = self.transform(imgx)
        if self.target_transform is not None:
            target = self.target_transform(target_base)
        # Note that the provided transform must have included a ToTensor
        imgs.append(imgx.unsqueeze(0))
        targets.append(target)
        # Create similar
        imgxp, targetp = img_base, target_base
        imgxp = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.contrast_transform is not None:
            imgxp = self.contrast_transform(imgxp)
        if self.target_transform is not None:
            targetp = self.target_transform(targetp)
        imgs.append(imgxp.unsqueeze(0))
        targets.append(targetp)
        # Create contrasted
        for i in range(self.k):
            randind = np.random.randint(len(self.data))
            img, targ = self.data[randind], int(self.labels[randind])
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                targ = self.target_transform(targ)
            imgs.append(img.unsqueeze(0))
            targets.append(targ)
        imgout = torch.cat(imgs, dim=0)
        return imgout, targets
        

# TODO
class Train_CURL():
    def __init__(self, svhn_path, augment=False, use_cuda=False):
        self.bulk = Net_Bulk()
        self.head = Net_Head()
        normalize = transforms.Compose(
                     [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ])
        augtrans = transforms.Compose(
                    [
                     transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
                     transforms.RandomAffine(20, scale=(0.9,1.1),shear=20, 
                                             resample=PIL.Image.BICUBIC, fillcolor=(100,100,100)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ])
        if augment:
            transform = augtrans
        else:
            transform = normalize
        trainset = ContrastedData(svhn_path, split='train', contrast_transform=augtrans, transform=transform, download=dload_dataset)
        self.trainset = trainset
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("CUDA not available")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        self.bulk.to(self.device)
        self.head.to(self.device)

    def trainbulk(self, batch_size=8, epochs=10, loss_freq=1):
        self.bulk.train()
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        optimizer = optim.SGD(self.bulk.parameters(), lr=0.001, momentum=0.9)
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

    def trainhead(self, batch_size=8, epochs=10, loss_freq=1):
        self.net.train()
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=2)
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

    testset = datasets.SVHN(svhn_path, split='test', transform=normalize, target_transform=None, download=dload_dataset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

    trainsup = Train_Sup(svhn_path, frac=0.1, shuffle=True, augment=True, use_cuda=True)
    trainsup.train(epochs=40)
