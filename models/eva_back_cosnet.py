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

    path='/home/yun13001/dataset/Carbon/tianyu_new_data/New_distribution/'
    dirnames = os.listdir(path)
    dirnames.remove('kinetic_curves.xlsx')
    dirnames=sorted(dirnames, key=lambda s: float(re.findall(r'\d+', s)[0]))
    print(dirnames)

    file_names={}
    for dirname in dirnames:
        for root, dirnames, filenames_x in os.walk(path+'/'+dirname+'/results/img'):
            break
        filenames_x = sorted(filenames_x, key=lambda s: float(re.findall(r'\d+', s)[2]))
        if dirname == '201' or dirname == '203':
            filenames_x.pop(0)
        else:
            filenames_x.pop(0)
            filenames_x.pop(0)

        file_names[dirname]=filenames_x

    train_name=['102_R1','102_R2', '302']
    val_name = ['103','301']
    #val_name = ['103']
    #val_name = ['301']
    test_name= ['201','203']
    #test_name=['201']
    #test_name=['203']
    all_name=['103','301','201','203']

    if names == 'val':
        names = val_name
    if names == 'test':
        names = test_name

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

    running_metric1.clear()
    running_metric2.clear()
    running_metric_all1.clear()
    running_metric_all2.clear()


    for name in names:

        running_metric1.clear()
        running_metric2.clear()

        ref=[]
        label1=[]
        label2=[]

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            #ref.append(x_ref)
            #label1.append(x_label1)
            #label2.append(x_label2)

            x_ref = np.array(x_ref)
            x_label1 = np.array(x_label1)
            x_label2 = np.array(x_label2)

            ref.append(x_ref)
            label1.append(x_label1)
            label2.append(x_label2)


        ref = np.array(ref)
        label1 = np.array(label1)
        label2 = np.array(label2)

        label1 = denormalize(label1)
        label2 = denormalize(label2)

        a1_f1=[]
        a2_f1=[]
        name_save=[]

        for m in range(len(ref)-1,-1,-1):

            pred_img1=[]
            pred_img2=[]

            numb=m

            name_ = imgs[m]
            print(name_)

            name_save.append(name_)
            if numb!=len(ref)-1:

                n=m
                name__ = imgs[n]

                img1 = ref[n]
                img2 = ref[len(ref)-1]

                #img1 = TF.to_tensor(img1).to(device)
                #img2 = TF.to_tensor(img2).to(device)
                #img1 = TF.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                #img2 = TF.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                #img1 = img1.unsqueeze(0)
                #img2 = img2.unsqueeze(0)

                meanval = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) 
                img1 = np.subtract(img1, np.array(meanval, dtype=np.float32))   
                img2 = np.subtract(img2, np.array(meanval, dtype=np.float32))  
                img1 = TF.to_tensor(img1).to(device)
                img2 = TF.to_tensor(img2).to(device)
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
                
                output_sum = 0 
                output = net_G(img1,img2) 
                print('output shape:',output[0].shape)

                output_sum = output_sum + output[0].data[0,0].cpu().numpy()
                mask = (output_sum*255).astype(np.uint8)
                mask = Image.fromarray(mask)

                try:
                   os.makedirs(res_path+'a1_b_order')
                except OSError:
                   pass

                try:
                   os.makedirs(res_path+'a1_b_order/'+name)
                except OSError:
                   pass

                vis_dir1 = res_path + 'a1_b_order/'+name+'/'
                
                path_save = vis_dir1+name_
                mask.save(path_save,'png')
                
