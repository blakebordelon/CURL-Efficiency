import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling
import spacy



TR_PATH ='aclImdb_v1.tar/aclimdb/train'

review = !cat {TRN}{trn_files[6]}
print(review[0])

spacy_tok = spacy.load('en')
