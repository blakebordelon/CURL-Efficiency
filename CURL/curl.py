"""
CURL - Contrastive Unsupervised Representation Learning
For use with images.
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
import sys

import matplotlib.pyplot as plt
import numpy as np

import PIL
from PIL import Image

##############################################################################################
# The methods below are for dynamically creating a subclass of the torchvision.datasets dataset
##############################################################################################
def _dataset_constructor(self, root,
                              split='train',
                              transform=None,
                              augment_transform=None,
                              C=10,
                              download=False,
                              k=1,
                              groundtruth=False
                              ):
    """This is used as the __init__ class for the generic extension to a dataset in
    torchvision.datasets.

    Parameters
    ----------
    root : str
        Path to the directory which contains the .mat dataset files (or is where
        you wish to download the dataset files)
    split : str
        Which part of the dataset to get (train or test)
    transform : torchvision.transforms transform
        Transform to apply to images
    augment_transform : torchvision.transforms transform
        Transform to apply for the purpose of data augmentation
    C : int
        The number of classes in the dataset.
    download : bool
        Whether to download the dataset if it's not found
    k : int
        How many negative samples (number of x-) for each x.
    groundtruth : bool
        Whether to sample x+ (x-) from the same class (different class) with perfect
        target information. If true, exact labels must have been passed with setlabels().
        If false, approximate labels must have been passed with setapproxlabels()

    Returns
    -------
    None
    """
    super(self.__class__, self).__init__(root,
                                         split=split,
                                         transform=transform,
                                         download=download)
    self.k = k
    self.C = C
    self.groundtruth = groundtruth
    self.byclass = None
    self.excludelabel = []  # List of numpy arrays containing acceptable classes to choose for class i
    if augment_transform is None:
        self.augment_transform = transform
    else:
        self.augment_transform = augment_transform
    if self.transform is None:
        print("Warning - a transform must be provided, at the very least transform.ToTensor()")
        print("After transformation, the result should be a tensor.")

def _dataset_setlabels(self, byclass, labeldict):
    """This is a function intended for the dynamic class enabling generic extensions
    to a dataset in torchvision.datasets. It saves a an exact classification
    of the unlabeled data to use for CURL.

    Parameters
    ----------
    byclass : list[list[int]]
        Should contain a list for every class, containing the dataset indices in that class.
    labeldict : dict(int)
        Given a dataset index, this dictionary returns the class label
        assigned to it.

    Returns
    -------
    None
    """
    self.byclass = byclass
    self.labeldict = labeldict
    for i in range(len(self.byclass)):
        if len(self.byclass[i]) == 0:
            # if there are no representatives in this class, we can't choose it when
            # looking for x-.
            exclusionlist.append(i) 
    exclusionlist = np.array(exclusionlist)
    for i in range(C):
        templist = list(range(C))
        templist.remove(i)
        templist = np.array(templist)
        templist = np.setdiff1d(templist, exclusionlist)
        self.excludelabel.append(templist)

def _dataset_setapproxlabels(self, approxbyclass, approxlabeldict):
    """This is a function intended for the dynamic class enabling generic extensions
    to a dataset in torchvision.datasets. It saves a an approximate classification
    of the unlabeled data to use for CURL.

    Parameters
    ----------
    approxbyclass : list[list[int]]
        Should contain a list for every class, containing the dataset indices predicted
        to be in that class.
    approxlabeldict : dict(int)
        Given a dataset index, this dictionary returns the approximate class label
        assigned to it.

    Returns
    -------
    None
    """
    self.approxbyclass = approxbyclass
    self.approxlabeldict = approxlabeldict
    exclusionlist = []
    for i in range(len(self.approxbyclass)):
        if len(self.approxbyclass[i]) == 0:
            # if there are no representatives in this class, we can't choose it when
            # looking for x-.
            exclusionlist.append(i) 
    exclusionlist = np.array(exclusionlist)
    for i in range(self.C):
        templist = list(range(self.C))
        templist.remove(i)
        templist = np.array(templist)
        templist = np.setdiff1d(templist, exclusionlist)
        self.excludelabel.append(templist)

def _dataset_getitem(self, index):
    """This is a function intended for the dynamic class enabling generic extensions
    to a dataset in torchvision.datasets. It gets the item in the dataset corresponding
    to the passed index

    Parameters
    ----------
    index : int
        The dataset index of the data point desired.

    Returns
    -------
    imgs : tensor of shape (k+2, X, Y)
        The x, x+, and k times x- images, as a stack of tensors (representing images)
    targets : tensor of shape (k+2)
        The targets (true labels) corresponding to x, x+, and k times x-.
    """
    imgs = []
    # The targets are provided, but in application CURL does not use these targets
    # because the labels are assumed unknown.
    targets = torch.zeros(self.k+2, dtype=torch.int64)
    
    # Get original data point (i.e. x)
    if self.labels is not None:
        img, target = self.data[index], int(self.labels[index])
    else:
        img, target = self.data[index], -1
    img = Image.fromarray(np.transpose(img, (1, 2, 0)))
    if self.augment_transform is not None:
        img = self.augment_transform(img)
    imgs.append(img.unsqueeze(0))
    targets[0] = target

    if self.groundtruth:
        label = target
        sim_index = np.random.choice(self.byclass[label])
    else:
        label = self.approxlabeldict[index]
        sim_index = np.random.choice(self.approxbyclass[label])

    # Get the similar datapoint (i.e. x+)
    if self.labels is not None:
        img, target = self.data[sim_index], int(self.labels[sim_index])
    else:
        img, target = self.data[sim_index], -1
    img = Image.fromarray(np.transpose(img, (1, 2, 0)))
    if self.augment_transform is not None:
        img = self.augment_transform(img)
    imgs.append(img.unsqueeze(0))
    targets[1] = target

    # Get the contrasting datapoints (i.e. the x-)
    for i in range(self.k):
        classnum = np.random.choice(self.excludelabel[label])
        if self.groundtruth:
            cont_index = np.random.choice(self.byclass[classnum])
        else:
            cont_index = np.random.choice(self.approxbyclass[classnum])
        if self.labels is not None:
            img, target = self.data[cont_index], int(self.labels[cont_index])
        else:
            img, target = self.data[cont_index], -1
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.augment_transform is not None:
            img = self.augment_transform(img)
        imgs.append(img.unsqueeze(0))
        targets[2+i] = target
    imgs = torch.cat(imgs, dim=0)
    return imgs, targets

def _dataset_getsingle(self, index):
    """This is a function intended for the dynamic class enabling generic extensions
    to a dataset in torchvision.datasets. It gets the item in the dataset corresponding
    to the passed index

    Parameters
    ----------
    index : int
        The dataset index of the data point desired.

    Returns
    -------
    img : tensor of shape (1, X, Y)
        The x tensor asked for (representing an image)
    target : int
        the target (label)
    """
    # Create original
    img, target = self.data[index], int(self.labels[index])
    img = Image.fromarray(np.transpose(img, (1, 2, 0)))
    if self.transform is not None:
        img = self.transform(img)
    # Note that the provided transform must have included a ToTensor
    return img.unsqueeze(0), target



class CURL():
    """Class for training with CURL.
    """
    def __init__(self, Dataset,
                       C,
                       dataset_path,
                       bulk,
                       head,
                       curlloss=None,
                       labeledfrac=0.5,
                       unlabeledfrac=0.5,
                       shuffle=True,
                       k=1,
                       transform=None,
                       augment_transform=None,
                       augment=False,
                       testsplit='test',
                       suptrainsplit='train',
                       curltrainsplit='train',
                       groundtruth=False,
                       labeled_indices=None,
                       unlabeled_indices=None,
                       combine=False,
                       use_cuda=True,
                       download_dataset=True):
        """Initialize the CURL training class

        Parameters
        ----------
        Dataset : torchvision.datasets dataset class
            A dataset from the torchvision.datasets module (like 
            torchvision.datasets.SVHN)
        C : int
            Number of classes in the target, e.g. 10 for CIFAR-10.
        dataset_path : str
            The path to the directory containing the dataset (.mat) files.
            If download_dataset is True and the files are not found, they
            will be downloaded to this directory.
        bulk : torch.nn.Module
            The bulk of the network
        head : torch.nn.Module
            The head of the network. The bulk feeds into the head to form the
            full classification network.
        curlloss : torch function
            The loss to use for the CURL portion of training. It should take in
            a batched (minibatch, k) tensor of f(x).f(x_k-) - f(x).f(x+) scalars.
        labeledfrac : float
            The fraction in [0,1] of the total training data to use as labeled
            data. unlabeledfrac + labeledfrac <= 1. Ignored if labeled_indices
            are provided.
        unlabeledfrac : float
            The fraction in [0,1] of the total training data to use as unlabeled
            data (i.e. ignore the class label). unlabeledfrac + labeledfrac <= 1.
            Ignored if unlabeled_indices are provided.
        shuffle : bool
            Whether to shuffle the order of the training set before selecting the
            labeled and unlabeled data. If False, the first fraction of images is
            used as labeled data, and the second fraction of images is used as unlabeled.
        k : int
            How many different-class points to use per forward pass of CURL, i.e. the
            number of x^-.
        transform : torchvision.transforms transformation
            The transformation used to turn the image into a tensor. Note that it must
            include transforms.ToTensor() somewhere. This parameter is required. Used
            for the test set as well.
        augment_transform : torchvision.transforms transformation
            The transformation used to turn the augment the image data (by applying
            transformations that shouldn't change the content of the image). If augment
            is True, this parameter must be passed. Note that augment_transform is used
            instead of transform during training.
        augment : bool
            Whether to use data augmentation through image transformations. If True,
            augment_transform must be provided.
        testsplit : str
            Where the data for test should come from. (typically 'test')
        suptrainsplit : str
            Where the data for supervised training should come from. (typically 'train')
        curltrainsplit : str
            Where the data for CURL training should come from. (typically 'train', unless the
            dataset comes with an 'unlabeled' split)
        groundtruth : bool
            Whether to use perfect in-class out-of-class sampling for CURL (CURL data must
            have labels).
        labeled_indices : list(int)
            A list of integers corresponding to which images to use during training as
            labeled images
        unlabeled_indices : list(int)
            A list of integers corresponding to which images to use during training as
            unlabeled images
        combine : bool
            Whether the labeled data should be contained as a subset of the unlabeled.
        use_cuda : bool
            Whether to use CUDA or the cpu.
        download_dataset : bool
            Whether to download the dataset if it is not found in dataset_path. If it is
            already there, it will be verified.

        Returns
        -------
        None
        """
        if labeledfrac + unlabeledfrac > 1.0:
            print("Labeled fraction plus unlabeled fraction cannot exceed 1")
            print("Setting to defaults")
            if labeledfrac <= 1.0:
                unlabeledfrac = 1.0 - labeledfrac
                print(f"Set unlabeled frac to {unlabeledfrac}")
            else:
                labeledfrac, unlabeledfrac = 0.5, 0.5
                print(f"Set labeled frac to {labeledfrac}")
                print(f"Set unlabeled frac to {unlabeledfrac}")

        self.labeledfrac = labeledfrac
        self.unlabeledfrac = unlabeledfrac

        self.Dataset = Dataset
        self.dataset_path = dataset_path
        self.bulk = bulk
        self.head = head
        self.curlloss = curlloss
        self.k = k
        self.C = C
        self.groundtruth = groundtruth

        self.transform = transform
        self.augment_transform = augment_transform
        self.augment = augment
        self.use_cuda = use_cuda

        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("CUDA not available")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        if not augment or self.augment_transform is None:
            self.augment_transform = transform

        self.suptrainset = self.Dataset(dataset_path, split=suptrainsplit, transform=self.augment_transform,
                                        download=download_dataset)
        self.testset = self.Dataset(dataset_path, split=testsplit, transform=transform,
                                    download=download_dataset)

        if labeled_indices is not None and unlabeled_indices is not None:
            self.labeled_indices = labeled_indices
            self.unlabeled_indices = unlabeled_indices
        elif labeled_indices is not None and unlabeled_indices is None:
            self.labeled_indices = labeled_indices
            trainset_size = len(self.suptrainset)
            indices = list(range(trainset_size))
            indices = np.setdiff1d(indices, labeled_indices)
            unlabeled_end = int(np.floor(unlabeledfrac * trainset_size))
            if shuffle:
                np.random.shuffle(indices)
            self.unlabeled_indices = indices[:unlabeled_end]
        elif labeled_indices is None and unlabeled_indices is not None:
            self.unlabeled_indices = unlabeled_indices
            trainset_size = len(self.suptrainset)
            indices = list(range(trainset_size))
            indices = np.setdiff1d(indices, unlabeled_indices)
            labeled_end = int(np.floor(labeledfrac * trainset_size))
            if shuffle:
                np.random.shuffle(indices)
            self.labeled_indices = indices[:labeled_end]
        else:
            trainset_size = len(self.suptrainset)
            indices = list(range(trainset_size))
            end = int(np.floor((unlabeledfrac+labeledfrac) * trainset_size))
            labeled_end = int(np.floor(labeledfrac/(labeledfrac + unlabeledfrac) * end))
            if shuffle:
                np.random.shuffle(indices)
            self.labeled_indices = indices[:labeled_end]
            if combine:
                self.unlabeled_indices = indices[:end]
            else:
                self.unlabeled_indices = indices[labeled_end:end]

        print(f"Number of labeled images: {len(self.labeled_indices)}")
        print(f"Number of unlabeled images: {len(self.unlabeled_indices)}")

        self.suptrain_sampler = SubsetRandomSampler(self.labeled_indices)
        self.curltrain_sampler = SubsetRandomSampler(self.unlabeled_indices)

        # Dynamically create the dataset subclass that will sample x, x+, x-
        ContrastedDataset = type("ContrastedDataset",
                                 (Dataset,),
                                 {"__init__": _dataset_constructor,
                                  "setlabels": _dataset_setlabels,
                                  "setapproxlabels": _dataset_setapproxlabels,
                                  "__getitem__": _dataset_getitem,
                                  "getsingle": _dataset_getsingle
                                  })
        self.curltrainset = ContrastedDataset(dataset_path, split=curltrainsplit,
                                              transform=self.transform,
                                              augment_transform=self.augment_transform,
                                              C=self.C,
                                              download=download_dataset,
                                              k=self.k,
                                              groundtruth=self.groundtruth
                                              )

        self.bulk.to(self.device)
        self.head.to(self.device)


    def get_approximate_labels(self):
        """Use the full network (bulk+head) in its current state to classify the unlabeled data.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.bulk.eval()
        self.head.eval()
        approxlabeldict = {}
        approxbyclass = []
        for i in range(self.C):
            approxbyclass.append([])
        t = tqdm(leave=True, total=len(self.unlabeled_indices))
        for i in self.unlabeled_indices:
            img, target = self.curltrainset.getsingle(i)
            img = img.to(self.device)
            output = self.bulk(img)
            output = self.head(output)
            approxlabel = torch.argmax(output, dim=-1, keepdim=False).squeeze(0).item()
            approxbyclass[approxlabel].append(i)
            approxlabeldict[i] = approxlabel
            t.update()
        t.close()
        self.curltrainset.setapproxlabels(approxbyclass, approxlabeldict)

    def get_exact_labels(self):
        """Use the labels to label the unlabeled data (only for ground truth testing)

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        labeldict = {}
        byclass = []
        for i in range(self.C):
            byclass.append([])
        t = tqdm(leave=True, total=len(self.unlabeled_indices))
        for i in self.unlabeled_indices:
            img, target = self.curltrainset.getsingle(i)
            byclass[target].append(i)
            labeldict[i] = target
            t.update()
        t.close()
        self.curltrainset.setlabels(byclass, labeldict)


    def curltrain(self, batch_size=5, epochs=10, loss_freq=100):
        """CURL training

        Parameters
        ----------
        batch_size : int
            Batch size
        epochs : int
            epochs to pass through
        loss_freq : int
            How often to compute the loss (i.e. calculate it every loss_freq iterations)

        Returns
        -------
        train_losses : list[float]
            List of training losses
        sims : list[float]
            Similarity term f(x)f(x+)
        conts : list[float]
            Contrast term f(x)f(x-)
        """
        self.bulk.train()
        trainloader = torch.utils.data.DataLoader(self.curltrainset, batch_size=batch_size, sampler=self.curltrain_sampler, num_workers=4)

        optimizer = optim.Adam(self.bulk.parameters(), lr=0.0001)
        train_losses = []
        sims = []
        conts = []
        prevsim = 0.
        prevcont = 0.
        bnum = 0
        t = tqdm(leave=True, total=epochs*len(trainloader))
        for epoch in range(epochs):
            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                # targets not used, this part is unsupervised
                inputs, targets = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                dims = inputs.shape
                # flatten x, x+, x- into the batch dimension for greater efficiency
                inputsflat = torch.flatten(inputs, start_dim=0, end_dim=1)
                outputsflat = self.bulk(inputsflat)
                # recover the (minibatch, k+2, d) structure.
                outputs = outputsflat.view(dims[0], dims[1], -1)
                
                # Hadamard product of f(x) and f(x+), and f(x) and all f(x-)
                # Then sum the last dimension, so we have computed f(x).f(x+), and k of f(x)f(x-)
                outputs2 = (outputs[:,0:1]*outputs[:,1:]).sum(-1)

                sim = outputs2[:,0:1]  # f(x).f(x+)

                contrast = outputs2[:,1:]  # all f(x).f(x-)

                minibatched_loss = self.curlloss(contrast - sim)

                loss = torch.mean(minibatched_loss)
                loss.backward()
                optimizer.step()

                # record loss
                loss_val = loss.cpu().item()
                if i % loss_freq == 0:
                    train_losses.append(loss_val)
                    prevsim = torch.mean(sim).cpu().item()
                    sims.append(prevsim)
                    prevcont = torch.mean(contrast).cpu().item()
                    conts.append(prevcont)
                bnum += 1
                t.update()
                t.set_postfix(sim=f'{prevsim:.2f}', cont=f'{prevcont:.2f}')
        t.close()
        self.bulk.eval()
        return train_losses, sims, conts


    def suptrain(self, batch_size=8, epochs=10, loss_freq=1, test_freq=1000):
        """Supervised training of just the head

        Parameters
        ----------
        batch_size : int
            Batch size
        epochs : int
            epochs to pass through
        loss_freq : int
            How often to compute the loss (i.e. calculate it every loss_freq iterations)
        test_freq : int
            How often to compute the test loss (i.e. calculate it every test_freq iterations)

        Returns
        -------
        train_losses : list[float]
            List of training losses
        test_accs : list[float]
            List of test accuracies
        """
        self.head.train()
        self.bulk.eval()
        trainloader = torch.utils.data.DataLoader(self.suptrainset, batch_size=batch_size, sampler=self.suptrain_sampler, num_workers=2)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=8, shuffle=False, num_workers=2)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.head.parameters(), lr=0.001)
        #optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        train_losses = []
        test_accs = []
        t = tqdm(leave=True, total=epochs*len(trainloader))
        bnum = 0
        for epoch in range(epochs):
            for i, data in enumerate(trainloader):

                if bnum % test_freq == 0:
                    correct = 0
                    total = 0
                    self.head.eval()
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
                    self.head.train()

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                with torch.no_grad():
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
        """Train full network with labeled data.

        Parameters
        ----------
        batch_size : int
            Batch size
        epochs : int
            epochs to pass through
        loss_freq : int
            How often to compute the loss (i.e. calculate it every loss_freq iterations)
        test_freq : int
            How often to compute the test loss (i.e. calculate it every test_freq iterations)

        Returns
        -------
        train_losses : list[float]
            List of training losses
        test_accs : list[float]
            List of test accuracies
        """
        self.bulk.train()
        self.head.train()
        trainloader = torch.utils.data.DataLoader(self.suptrainset, batch_size=batch_size, sampler=self.suptrain_sampler, num_workers=2)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=8, shuffle=False, num_workers=2)

        criterion = nn.NLLLoss()
        #optimizer = optim.SGD(list(self.bulk.parameters()) + list(self.head.parameters()), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(list(self.bulk.parameters()) + list(self.head.parameters()), lr=0.001)
        train_losses = []
        test_accs = []
        bnum = 0
        t = tqdm(leave=True, total=epochs*len(trainloader))
        for epoch in range(epochs):
            for i, data in enumerate(trainloader):

                if bnum % test_freq == 0:
                    correct = 0
                    total = 0
                    self.bulk.eval()
                    self.head.eval()
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
                self.bulk.train()
                self.head.train()

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
