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
def _datasetconstructor(self, root,
                              split='train',
                              transform=None,
                              target_transform=None,
                              download=False,
                              k=1,
                              groundtruth=False
                              ):
    super(self.__class__, self).__init__(root,
                                         split=split,
                                         transform=transform,
                                         target_transfrom=target_transform,
                                         download=download)
    self.k = k
    self.groundtruth = groundtruth
    self.byclass = None
    self.excludelabel = []
    if transform is None:
        print("Warning - a transform must be provided, at the very least transform.ToTensor()")
        print("After transformation, the result should be a tensor.")


class CURL():
    """Class for training with CURL.
    """
    def __init__(self, dataset,
                       C,
                       dataset_path,
                       bulk,
                       head,
                       curlloss=None,
                       labeledfrac=0.5,
                       unlabeledfrac=0.5,
                       shuffle=True,
                       j=1,
                       k=1,
                       transform=None,
                       augment_transform=None,
                       augment=False,
                       labeled_indices=None,
                       unlabeled_indices=None,
                       use_cuda=True,
                       download_dataset=True):
        """Initialize the CURL training class

        Parameters
        ----------
        dataset : torchvision.datasets dataset
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
            The loss to use for the CURL portion of training. It should have the
            signature loss( sum_sim, sum_diff ) where sum_sim = Sum( f(x).f(x+) )
            and sum_diff = Sum( f(x).f(x-) ).
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
        labeled_indices : list(int)
            A list of integers corresponding to which images to use during training as
            labeled images
        unlabeled_indices : list(int)
            A list of integers corresponding to which images to use during training as
            unlabeled images
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
            labeledfrac, unlabeledfrac = 0.5, 0.5
        self.labeledfrac = labeledfrac
        self.unlabeledfrac = unlabeledfrac

        self.dataset = dataset
        self.dataset_path = dataset_path
        self.bulk = bulk
        self.head = head
        self.curlloss = curlloss
        self.k = k
        self.C = C

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

        # TODO
        self.suptrainset = datasets.SVHN(svhn_path, split='train', transform=transform, target_transform=None, download=dload_dataset)
        self.testset = datasets.SVHN(svhn_path, split='test', transform=normalize, target_transform=None, download=dload_dataset)

        if labeled_indices is not None and unlabeled_indices is not None:
            self.labeled_indices = labeled_indices
            self.unlabeled_indices = unlabeled_indices
        elif labeled_indices is not None and unlabeled_indices is None:
            self.labeled_indices = labeled_indices
            trainset_size = len(self.trainset)
            indices = list(range(trainset_size))
            indices = np.setdiff1d(indices, labeled_indices)
            unlabeled_end = int(np.floor(unlabeledfrac * trainset_size))
            if shuffle:
                np.random.shuffle(indices)
            self.unlabeled_indices = indices[:unlabeled_end]
        elif labeled_indices is None and unlabeled_indices is not None:
            self.unlabeled_indices = unlabeled_indices
            trainset_size = len(self.trainset)
            indices = list(range(trainset_size))
            indices = np.setdiff1d(indices, unlabeled_indices)
            labeled_end = int(np.floor(labeledfrac * trainset_size))
            if shuffle:
                np.random.shuffle(indices)
            self.labeled_indices = indices[:labeled_end]
        else:
            trainset_size = len(self.trainset)
            indices = list(range(trainset_size))
            end = int(np.floor((unlabeledfrac+labeledfrac) * trainset_size))
            labeled_end = int(np.floor(labeledfrac/(labeledfrac + unlabeledfrac) * end))
            if shuffle:
                np.random.shuffle(indices)
            self.labeled_indices = indices[:labeled_end]
            self.unlabeled_indices = indices[labeled_end:end]

        print(f"Number of labeled images: {len(self.labeled_indices)}")
        print(f"Number of unlabeled images: {len(self.unlabeled_indices)}")

        self.suptrain_sampler = SubsetRandomSampler(labeled_indices)
        self.curltrain_sampler = SubsetRandomSampler(unlabeled_indices)

        #self.curltrainset = ContrastedData(svhn_path, split='train', accepted_indices=curltrain_indices, contrast_transform=contrasttrans, k=k, transform=transform, download=dload_dataset)
        self.curltrainset = ApproxContrastedData(svhn_path, split='train', contrast_transform=contrasttrans, k=k, transform=normalize, download=dload_dataset)

        self.approxclasses = []
        for i in range(C):
            self.approxclasses.append([])
        self.bulk.to(self.device)
        self.head.to(self.device)


