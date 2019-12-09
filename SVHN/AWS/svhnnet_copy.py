""" Basic CNN network for testing CURL on the SVHN dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import time

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
        x = torch.tanh(self.fc1(x))
        return x

class Net_Head(nn.Module):
    def __init__(self):
        super(Net_Head, self).__init__()
        self.fc2 = nn.Linear(120,40)
        self.fc3 = nn.Linear(40,10)
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
    def __init__(self, svhn_path, frac=0.5, shuffle=True, augment=True, use_cuda=False, dload_dataset=False):
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
                     transforms.RandomApply(augaffine, p=0.8),
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
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, sampler=self.train_sampler, num_workers=4)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=8, shuffle=False, num_workers=4)

        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        train_losses = []
        bnum = 0
        t = tqdm(leave=True, total=epochs*len(trainloader))
        for epoch in range(epochs):
            for i, data in enumerate(trainloader):

                if bnum % test_freq == 0:
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data[0].to(self.device), data[1].to(self.device)
                            outputs = self.net(images)
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
    def __init__(self, root, split='train', accepted_indices=None, contrast_transform=None, k=1, 
                 transform=None, target_transform=None, download=False):
        super(ContrastedData, self).__init__(root, split=split, transform=transform,  
                                             target_transform=target_transform, download=download)
        self.contrast_transform = contrast_transform
        self.k = k
        self.accepted_indices = accepted_indices
        if transform is None:
            print("Warning - a transform must be provided, at the very least transform.ToTensor()")
            print("After transformation, the result should be a tensor.")

    def __getitem__(self, index):
        imgs = []
        targets = torch.zeros(self.k+2, dtype=torch.int64)  # We actually shouldn't be using the target for CURL anyways
        # Create original
        img_base, target_base = self.data[index], int(self.labels[index])
        target = target_base
        imgx = Image.fromarray(np.transpose(img_base, (1, 2, 0)))
        if self.transform is not None:
            imgx = self.transform(imgx)
        if self.target_transform is not None:
            target = self.target_transform(target_base)
        # Note that the provided transform must have included a ToTensor
        imgs.append(imgx.unsqueeze(0))
        targets[0] = target
        # Create similar
        imgxp, targetp = img_base, target_base
        imgxp = Image.fromarray(np.transpose(imgxp, (1, 2, 0)))
        if self.contrast_transform is not None:
            imgxp = self.contrast_transform(imgxp)
        if self.target_transform is not None:
            targetp = self.target_transform(targetp)
        imgs.append(imgxp.unsqueeze(0))
        targets[1] = targetp
        # Create contrasted
        for i in range(self.k):
            if self.accepted_indices is not None:
                randind = np.random.choice(self.accepted_indices)
            else:
                randind = np.random.randint(len(self.data))
            img, targ = self.data[randind], int(self.labels[randind])
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                targ = self.target_transform(targ)
            imgs.append(img.unsqueeze(0))
            targets[2+i] = targ
        imgout = torch.cat(imgs, dim=0)
        return imgout, targets


class ApproxContrastedData(datasets.SVHN):
    def __init__(self, root, split='train', contrast_transform=None, k=1, 
                 transform=None, target_transform=None, download=False):
        super(ApproxContrastedData, self).__init__(root, split=split, transform=transform,  
                                             target_transform=target_transform, download=download)
        self.contrast_transform = contrast_transform
        self.k = k
        self.byclass = None
        self.excludelabel = []
        for i in range(10):
            templist = list(range(10))
            templist.remove(i)
            self.excludelabel.append(templist)
        if transform is None:
            print("Warning - a transform must be provided, at the very least transform.ToTensor()")
            print("After transformation, the result should be a tensor.")

    def setapproxlabels(self, byclass, labeldict):
        self.byclass = byclass
        self.labeldict = labeldict


    def __getitem__(self, index):
        imgs = []
        targets = torch.zeros(self.k+2, dtype=torch.int64)  # We actually shouldn't be using the target for CURL anyways
        # Create original
        img_base, target_base = self.data[index], int(self.labels[index])
        target = target_base
        imgx = Image.fromarray(np.transpose(img_base, (1, 2, 0)))
        if self.transform is not None:
            imgx = self.transform(imgx)
        if self.target_transform is not None:
            target = self.target_transform(target_base)
        # Note that the provided transform must have included a ToTensor
        imgs.append(imgx.unsqueeze(0))
        targets[0] = target

        approxlabel = self.labeldict[index]
        # Create similar
        simind = np.random.choice(self.byclass[approxlabel])
        img, targ = self.data[simind], int(self.labels[simind])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.contrast_transform(img)
        if self.target_transform is not None:
            targ = self.target_transform(targ)
        imgs.append(img.unsqueeze(0))
        targets[1] = targ

        # Create contrasted
        for i in range(self.k):
            classnum = np.random.choice(self.excludelabel[approxlabel])
            randind = np.random.choice(self.byclass[classnum])
            img, targ = self.data[randind], int(self.labels[randind])
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                targ = self.target_transform(targ)
            imgs.append(img.unsqueeze(0))
            targets[2+i] = targ
        imgout = torch.cat(imgs, dim=0)
        return imgout, targets

    def getsingle(self, index):
        # Create original
        img_base, target_base = self.data[index], int(self.labels[index])
        target = target_base
        imgx = Image.fromarray(np.transpose(img_base, (1, 2, 0)))
        if self.transform is not None:
            imgx = self.transform(imgx)
        if self.target_transform is not None:
            target = self.target_transform(target_base)
        # Note that the provided transform must have included a ToTensor
        return imgx.unsqueeze(0), target
        

# TODO
class Train_CURL():
    def __init__(self, svhn_path, curlfrac=0.5, supfrac=0.5, k=1, shuffle=True, augment=False, use_cuda=False, dload_dataset=False):
        self.k = k
        self.softplus = nn.Softplus()
        self.bulk = Net_Bulk()
        self.head = Net_Head()
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
                     transforms.RandomApply(augaffine, p=0.8),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ])
        contrasttrans = transforms.Compose(
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
        self.suptrainset = datasets.SVHN(svhn_path, split='train', transform=transform, target_transform=None, download=dload_dataset)
        self.testset = datasets.SVHN(svhn_path, split='test', transform=normalize, target_transform=None, download=dload_dataset)

        if curlfrac + supfrac > 1.0:
            print("CURL fraction plus SUP fraction cannot exceed 1")
            print("Setting to defaults")
            curlfrac, supfrac = 0.5, 0.5
        trainset_size = len(self.suptrainset)
        indices = list(range(trainset_size))
        end = int(np.floor((curlfrac+supfrac) * trainset_size))
        curlend = int(np.floor(curlfrac/(supfrac + curlfrac) * end))
        if shuffle:
            np.random.shuffle(indices)
        curltrain_indices = indices[:curlend]
        suptrain_indices = indices[curlend:end]

        self.curltrain_indices = curltrain_indices
        print(f"Number of labeled images: {len(suptrain_indices)}")
        print(f"Number of unlabeled images: {len(curltrain_indices)}")
        self.suptrain_sampler = SubsetRandomSampler(suptrain_indices)
        self.curltrain_sampler = SubsetRandomSampler(curltrain_indices)

        #self.curltrainset = ContrastedData(svhn_path, split='train', accepted_indices=curltrain_indices, contrast_transform=contrasttrans, k=k, transform=transform, download=dload_dataset)
        self.curltrainset = ApproxContrastedData(svhn_path, split='train', contrast_transform=contrasttrans, k=k, transform=normalize, download=dload_dataset)

        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("CUDA not available")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.approxclasses = []
        for i in range(10):
            self.approxclasses.append([])
        self.bulk.to(self.device)
        self.head.to(self.device)


    def approx_labels(self):
        self.bulk.eval()
        self.head.eval()
        self.labeldict = {}
        t = tqdm(leave=True, total=len(self.curltrain_indices))
        for i in self.curltrain_indices:
            img, target = self.curltrainset.getsingle(i)
            img = img.to(self.device)
            output = self.bulk(img)
            output = self.head(output)
            approxlabel = torch.argmax(output, dim=-1, keepdim=False).squeeze(0).item()
            self.approxclasses[approxlabel].append(i)
            self.labeldict[i] = approxlabel
            t.update()
        t.close()
        self.curltrainset.setapproxlabels(self.approxclasses, self.labeldict)


    def curltrain(self, batch_size=8, epochs=10, loss_freq=1):
        self.approx_labels()

        self.bulk.train()
        trainloader = torch.utils.data.DataLoader(self.curltrainset, batch_size=batch_size, sampler=self.curltrain_sampler, num_workers=4)

        #optimizer = optim.SGD(self.bulk.parameters(), lr=0.001, momentum=0.3)
        optimizer = optim.Adam(self.bulk.parameters(), lr=0.0001)
        train_losses = []
        bnum = 0
        avgdist = torch.tensor(0., device=self.device)
        t = tqdm(leave=True, total=epochs*len(trainloader))
        for epoch in range(epochs):
            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                # labels not used, this part is unsupervised
                inputs, targets = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                dims = inputs.shape
                inputsflat = torch.flatten(inputs, start_dim=0, end_dim=1)
                outputsflat = self.bulk(inputsflat)
                outputs = outputsflat.view(dims[0], dims[1], -1)

                # # Get approximated target values
                # with torch.no_grad():
                #     classflat = bulkcopy(inputsflat)  # Original bulk after sup train
                #     classflat = self.head(classflat)
                #     classflat = torch.argmax(classflat, dim=-1, keepdim=False)
                #     approxtargets = classflat.view(dims[0], dims[1])

                # Hadamard product of f and f+, and f and all f-
                # Then sum the last dimension, so we have computed f^t f+, and k of f^t f-
                outputs2 = (outputs[:,0:1]*outputs[:,1:]).sum(-1)
                sim = outputs2[:,0]  # f^t f+
                contrast = outputs2[:,1:].sum(-1)  # sum of all f^t f-

                #contrast = (outputs2[:,1:]*torch.where(approxtargets[:,2:]==approxtargets[:,0].unsqueeze(-1),torch.tensor(-1.,device = self.device),torch.tensor(1., device=self.device))).sum(-1)  # sum of all f^t f-

                #contrast = (outputs2[:,1:]*torch.where(targets[:,2:]==targets[:,0],torch.tensor(0.,device = self.device),torch.tensor(1., device=self.device))).sum(-1)  # sum of all f^t f-
                #sim = sim + (outputs2[:,1:]*torch.where(targets[:,2:]==targets[:,0],torch.tensor(1.,device = self.device),torch.tensor(0., device=self.device))).sum(-1)  # sum of all f^t f-

                #print(f'contrast {contrast/self.k}')
                #print(f'sim {sim}')
                # if bnum % 100 == 0:
                #     plt.scatter(bnum,torch.mean((contrast/self.k).detach().cpu()).numpy(), c='r')
                #     plt.scatter(bnum,torch.mean(sim.detach().cpu()).numpy(), c='b')
                #     plt.pause(0.1)
                minibatched_loss = self.softplus((contrast - sim)/outputs.shape[-1])

                loss = torch.mean(minibatched_loss)
                loss.backward()
                optimizer.step()

                # record loss
                loss_val = loss.cpu().item()
                if i % loss_freq == 0:
                    train_losses.append(loss_val)
                bnum += 1
                t.update()
                t.set_postfix(epoch=f'{epoch}/{epochs-1}', loss=f'{loss_val:.2e}')
        t.close()
        self.bulk.eval()
        #plt.show()
        return train_losses

    def suptrain(self, batch_size=8, epochs=10, loss_freq=1, test_freq=1000):
        self.head.train()
        trainloader = torch.utils.data.DataLoader(self.suptrainset, batch_size=batch_size, sampler=self.suptrain_sampler, num_workers=4)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=8, shuffle=False, num_workers=4)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.head.parameters(), lr=0.001, momentum=0.9)
        train_losses = []
        test_accs = []
        t = tqdm(leave=True, total=epochs*len(trainloader))
        bnum = 0
        for epoch in range(epochs):
            for i, data in enumerate(trainloader):

                if bnum % test_freq == 0:
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data[0].to(self.device), data[1].to(self.device)
                            outputs = self.bulk(images)
                            outputs = self.head(outputs)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    test_accs.append(100 * correct/total)
                    print(f'Epoch {epoch}: test accuracy = {100 * correct/total}')

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.bulk(inputs)
                outputs = self.head(outputs)
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
        self.head.eval()
        return train_losses, test_accs

    def train(self, batch_size=8, epochs=10, loss_freq=1, test_freq=10000):
        self.bulk.train()
        self.head.train()
        trainloader = torch.utils.data.DataLoader(self.suptrainset, batch_size=batch_size, sampler=self.suptrain_sampler, num_workers=4)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=8, shuffle=False, num_workers=4)

        criterion = nn.NLLLoss()
        optimizer = optim.SGD(list(self.bulk.parameters()) + list(self.head.parameters()), lr=0.001, momentum=0.9)
        train_losses = []
        test_accs = []
        bnum = 0
        t = tqdm(leave=True, total=epochs*len(trainloader))
        for epoch in range(epochs):
            for i, data in enumerate(trainloader):

                if bnum % test_freq == 0:
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data[0].to(self.device), data[1].to(self.device)
                            outputs = self.bulk(images)
                            outputs = self.head(outputs)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    test_accs.append(100 * correct/total)
                    print(f'Epoch {epoch}: test accuracy = {100 * correct/total}')

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.bulk(inputs)
                outputs = self.head(outputs)
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
        self.bulk.eval()
        self.head.eval()
        return train_losses, test_accs


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

    #trainsup = Train_Sup(svhn_path, frac=0.01, shuffle=True, augment=True, use_cuda=True)
    #trainsup.train(epochs=100, test_freq=2000)
    traincurl = Train_CURL(svhn_path, curlfrac=0.04, supfrac=0.0015, k=5, shuffle=True, augment=True, use_cuda=True)
    #for i in range(1):
    #    traincurl.train(epochs=30, batch_size=5,  test_freq=2000)
    #    traincurl.curltrain(epochs=1, batch_size=5)

    #_, sup_acc = traincurl.train(epochs=150, batch_size=5,  test_freq=1000)
    #traincurl.curltrain(epochs=1, batch_size=5)
    #_, postcurl_acc = traincurl.suptrain(epochs=10, batch_size=5,  test_freq=1000)

    _, sup_acc = traincurl.train(epochs=1000, batch_size=5,  test_freq=1000)
    traincurl.curltrain(epochs=200, batch_size=5)
    _, postcurl_acc = traincurl.suptrain(epochs=1000, batch_size=5,  test_freq=1000)

    supfile = "plots/aws-0015.npy"
    curlfile = "plots/aws-04-0015.npy"
    sup_arr = np.array(sup_acc)
    np.save(supfile, sup_arr)
    curl_arr = np.array(postcurl_acc)
    np.save(curlfile, curl_arr)
