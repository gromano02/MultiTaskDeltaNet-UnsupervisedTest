import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import re
from PIL import Image
import pickle
import cv2
import scipy.misc
import sys
import random
import timeit
from collections import OrderedDict

import utils
from utils import de_norm
from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

from siamese_model_conf import CoattentionNet

def normalize(input_data):
    return (input_data.astype(np.float32))/255.0

def denormalize(input_data):
    input_data = (input_data) * 255
    return input_data.astype(np.uint8)

palette = np.array([[1, 1, 1],  # "no change"
                    [0.5, 0, 0],  # "appearing(1)"
                    [0, 0.5, 0],  # "disappearing(-1)"
                    [0.4,0.4,0.5]], dtype='float32')  # "overlap(2)"  [0.4,0.4,0.5]

