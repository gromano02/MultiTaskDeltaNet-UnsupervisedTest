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

from utils import de_norm
import utils
#from sklearn.metrics import confusion_matrix
import seaborn as sns
from misc.metric_tool import ConfuseMatrixMeter

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

from models.cosnet.siamese_model_conf import CoattentionNet

def normalize(input_data):
    return (input_data.astype(np.float32))/255.0

def denormalize(input_data):
    input_data = (input_data) * 255
    return input_data.astype(np.uint8)

palette = np.array([[1, 1, 1],  # "no change"
                    [0.5, 0, 0],  # "appearing(1)"
                    [0, 0.5, 0],  # "disappearing(-1)"
                    [0.4,0.4,0.5]], dtype='float32')  # "overlap(2)"  [0.4,0.4,0.5]


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
       You probably saved the model using nn.DataParallel, which stores the model in module, and now you are trying to load it
       without DataParallel. You can either add a nn.DataParallel temporarily in your network for loading purposes, or you can
       load the weights file, create a new ordered dict without the module prefix, and load it back
    """
    state_dict_new = OrderedDict()
    #print(type(state_dict))
    for k, v in state_dict.items():
        #print(k)
        name = k[7:] # remove the prefix module.
        # My heart is broken, the pytorch have no ability to do with the problem.
        state_dict_new[name] = v
        if name == 'linear_e.weight':
            np.save('weight_matrix.npy',v.cpu().numpy())
    return state_dict_new

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))#定义一个sigmoid方法，其本质就是1/(1+e^-x)


def cosnetEva(model_path,res_path,names):  ### name:val or test

    net_G = CoattentionNet(num_classes=2)

    running_metric1 = ConfuseMatrixMeter(n_class=2)     ###### each dataset
    running_metric2 = ConfuseMatrixMeter(n_class=2)

    running_metric_all1 = ConfuseMatrixMeter(n_class=2)     ###### all dataset
    running_metric_all2 = ConfuseMatrixMeter(n_class=2)

    PATH = model_path+'co_attention.pth'

    saved_state_dict = torch.load(PATH, map_location=lambda storage, loc: storage)
    net_G.load_state_dict( convert_state_dict(saved_state_dict["model"]) )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G = nn.DataParallel(net_G)
    net_G.to(device)
    net_G.eval()


    path='/home/yun13001/dataset/Carbon/FBMS/Testset/cars1/'

    for root, dirnames, filenames_x in os.walk(path):
        break
    
    #filenames_x = sorted(filenames_x, key=lambda s: float(re.findall(r'\d+', s)))
    filenames_x = sorted(filenames_x, key=lambda s: tuple(map(int, re.findall(r'\d+', s))))
    filenames_x.pop(0)
    print(filenames_x)

    
    ref=[]
    #for i in range(len(filenames_x)):
    for i in filenames_x:
        x_ref = Image.open(path+i).convert('RGB')
        x_ref = np.array(x_ref) 
        ref.append(x_ref)
     
    ref = np.array(ref) 

    for m in range(len(ref)-1,-1,-1):

        pred_img1=[]
        pred_img2=[]

        numb=m

        name_ = filenames_x[m]
        print(name_)

        n=m            
        img1 = ref[n]
        img2 = ref[len(ref)-1]    

        meanval = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
        img1 = np.subtract(img1, np.array(meanval, dtype=np.float32))
        img2 = np.subtract(img2, np.array(meanval, dtype=np.float32))
        img1 = TF.to_tensor(img1).to(device)
        img2 = TF.to_tensor(img2).to(device)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)        


        output_sum = 0
        output = net_G(img1,img2)
        #print('output shape:',output[0].shape)

        output_sum = output_sum + output[0].data[0,0].cpu().numpy()
        mask = (output_sum*255).astype(np.uint8)
        mask = Image.fromarray(mask)

        try:
            os.makedirs(res_path+'fbms_b_order')
        except OSError:
            pass

        try:
            os.makedirs(res_path+'fbms_b_order/'+'cars1')
        except OSError:
            pass

        vis_dir1 = res_path + 'fbms_b_order/'+'cars1'+'/'

        path_save = vis_dir1+name_
        mask.save(path_save,'png')
        
