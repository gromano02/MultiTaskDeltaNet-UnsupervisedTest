import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils

from datasets.CD_dataset import CDDataset


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    #print('test11:',tensor_data.shape)
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    #print('test22:',vis.shape)
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis

def make_numpy_grid_lb(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    #print('test1:',tensor_data.shape)
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    #print('test1_1:',vis.shape)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    #print('test2:',vis.shape)
    #if vis.shape[2] == 1:
    #    vis = np.stack([vis, vis, vis], axis=-1)
    return vis[:,:,0]

def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
