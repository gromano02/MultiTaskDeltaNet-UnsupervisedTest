import numpy as np
import matplotlib.pyplot as plt
import os
import re
from PIL import Image
from monai.metrics import compute_hausdorff_distance, HausdorffDistanceMetric

from models.nets import *
from models.sia_two_out_four_class import Siamese
#from models.siamese_with_weights import Siamese
from misc.metric_tool import ConfuseMatrixMeter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils import de_norm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import utils


def normalize(input_data):
    return (input_data.astype(np.float32))/255.0

def denormalize(input_data):
    input_data = (input_data) * 255
    return input_data.astype(np.uint8)

palette = np.array([[1, 1, 1],  # "no change"
                    [0.5, 0, 0],  # "appearing(1)"
                    [0, 0.5, 0],  # "disappearing(-1)"
                    [0.4,0.4,0.5]], dtype='float32')  # "overlap(2)"  [0.4,0.4,0.5]

def binary_to_one_hot(binary_image):
    """Converts a binary image to one-hot encoding."""

    # Ensure the image is binary
    if binary_image.max() > 1:
        binary_image = (binary_image > 0).astype(int)

    # Create the one-hot encoded array
    one_hot = np.zeros((binary_image.shape[0], binary_image.shape[1], 2))

    # Fill in the one-hot values
    one_hot[:, :, 0] = 1 - binary_image
    one_hot[:, :, 1] = binary_image

    return one_hot

def CDEva(model_path,res_path,names):

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

    if names == 'val':
        names = val_name
    if names == 'test':
        names = test_name

    net_G = Siamese(in_channels=3, out_channels=4, init_features=32)


    PATH = model_path+'best_m.pt'
    #PATH = model_path+'best_a1.pt'
    checkpoint = torch.load(PATH)
    net_G.load_state_dict(checkpoint['model_G_state_dict'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G = nn.DataParallel(net_G)
    net_G.to(device)
    net_G.eval()

    #### overall hd
    a1_hd=[]
    a2_hd=[]
    pred_img1=[]
    pred_img2=[]
    tar_img1 =[]
    tar_img2 =[]

    for name in names:

        ref=[]
        label1=[]
        label2=[]

        print('name:',name)

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

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

        #### case hd:
        a11_hd=[]
        a22_hd=[]
        pred_img11=[]
        pred_img22=[]
        tar_img11 =[]
        tar_img22 =[]
        name_save=[]

        for m in range(len(ref)):

            predd_img1=[]  #### save pred image for average (ensemble)
            predd_img2=[]

            numb=m
            name_ = imgs[m]

            name_save.append(name_)

            for n in range(len(ref)):
                name1 = imgs[n]
                name2 = imgs[n]

                img1 = ref[m]
                img2 = ref[n]
                img1 = TF.to_tensor(img1).to(device)
                img2 = TF.to_tensor(img2).to(device)
                img1 = TF.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                img2 = TF.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)

                G_pred1, G_pred2 = net_G(img1, img2)

                trans = nn.Softmax(dim=1)
                G_pred1 = trans(G_pred1)
                G_pred2 = trans(G_pred2)

                pred1 = utils.make_numpy_grid(G_pred1)
                pred2 = utils.make_numpy_grid(G_pred2)

                pred1 = np.argmax(pred1, axis=2)
                pred2 = np.argmax(pred2, axis=2)


                ##### a1
                for l in range(pred1.shape[0]):
                    for k in range(pred1.shape[1]):

                        if pred1[l,k]==0:
                            pred1[l,k]=1

                        if pred1[l,k]==2:
                            pred1[l,k]=0
                        if pred1[l,k]==3:
                            pred1[l,k]=0

                pred1 = binary_to_one_hot(pred1)
                pred1 = np.transpose(pred1,(2,0,1))
                predd_img1.append(pred1)


                ##### a2
                for l in range(pred2.shape[0]):
                    for k in range(pred2.shape[1]):

                        if pred2[l,k]==0:
                            pred2[l,k]=1

                        if pred2[l,k]==2:
                            pred2[l,k]=0
                        if pred2[l,k]==3:
                            pred2[l,k]=0

                pred2 = binary_to_one_hot(pred2)
                pred2 = np.transpose(pred2,(2,0,1))
                predd_img2.append(pred2)

            ##### a1 average
            pred1 = sum(predd_img1)
            num = len(predd_img1)
            pred1 = pred1/num
            pred1[pred1>=0.5]=1.0
            pred1[pred1<0.5]=0.0                

            pred_img11.append(pred1)
            pred_img1.append(pred1)

            target1 = label1[m]
            target1 = target1[:,:,0]
            target1 = binary_to_one_hot(target1)
            target1 = np.transpose(target1,(2,0,1))
            tar_img11.append(target1)
            tar_img1.append(target1)            

            ##### a2 average
            pred2 = sum(predd_img2)
            num = len(predd_img2)
            pred2 = pred2/num
            pred2[pred2>=0.5]=1.0
            pred2[pred2<0.5]=0.0

            pred_img22.append(pred2)
            pred_img2.append(pred2)

            target2 = label2[m]
            target2 = target2[:,:,0]
            target2 = binary_to_one_hot(target2)
            target2 = np.transpose(target2,(2,0,1))
            tar_img22.append(target2)
            tar_img2.append(target2)

        ##### a1 HD
        pred_img11= np.array(pred_img11)
        tar_img11 = np.array(tar_img11)

        pred_img11 = torch.tensor(pred_img11)
        tar_img11 = torch.tensor(tar_img11)

        pred_img11 = pred_img11.to(torch.int64)
        tar_img11 = tar_img11.to(torch.int64)

        metric = HausdorffDistanceMetric(include_background=False, percentile=95)
        hd95 = metric(pred_img11, tar_img11)
        print('a1:',hd95)

        hd_img1 =hd95.tolist()
        hd_img1 = np.array(hd_img1)
        name_save = np.array(name_save)

        hd_img1 = np.flip(hd_img1)
        name_save = np.flip(name_save)

        combine1 = np.array([[a,str(b)] for a, b in zip(name_save, hd_img1)])

        try:
            os.makedirs(res_path+'a1_ensamble')
        except OSError:
            pass

        try:
            os.makedirs(res_path+'a1_ensamble/'+name)
        except OSError:
            pass

        vis_dir1 = res_path + 'a1_ensamble/'+name
        np.savetxt(vis_dir1+'/'+'a1_hd_'+name+".txt", combine1, fmt="%s", delimiter=",")

        hd95 = torch.mean(hd95)

        print('name:',name)
        print('a1 hd95:',hd95)


        ##### a2 HD
        pred_img22= np.array(pred_img22)
        tar_img22 = np.array(tar_img22)

        pred_img22 = torch.tensor(pred_img22)
        tar_img22 = torch.tensor(tar_img22)

        pred_img22 = pred_img22.to(torch.int64)
        tar_img22 = tar_img22.to(torch.int64)

        metric = HausdorffDistanceMetric(include_background=False, percentile=95)
        hd95 = metric(pred_img22, tar_img22)
        print('a2:',hd95)


        hd_img2 =hd95.tolist()
        hd_img2 = np.array(hd_img2)

        hd_img2 = np.flip(hd_img2)

        combine2 = np.array([[a,str(b)] for a, b in zip(name_save, hd_img2)])

        try:
            os.makedirs(res_path+'a2_ensamble')
        except OSError:
            pass

        try:
            os.makedirs(res_path+'a2_ensamble/'+name)
        except OSError:
            pass

        vis_dir2 = res_path + 'a2_ensamble/'+name
        np.savetxt(vis_dir2+'/'+'a2_hd_'+name+".txt", combine2, fmt="%s", delimiter=",")

        hd95 = torch.mean(hd95)

        print('name:',name)
        print('a2 hd95:',hd95)


    pred_img1= np.array(pred_img1)
    tar_img1 = np.array(tar_img1)

    pred_img1 = torch.tensor(pred_img1)
    tar_img1 = torch.tensor(tar_img1)

    pred_img1 = pred_img1.to(torch.int64)
    tar_img1 = tar_img1.to(torch.int64)

    metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    hd95 = metric(pred_img1, tar_img1)
    hd95 = torch.mean(hd95)

    print('name:',names)
    print('a1 hd95:',hd95)


    pred_img2= np.array(pred_img2)
    tar_img2 = np.array(tar_img2)

    pred_img2 = torch.tensor(pred_img2)
    tar_img2 = torch.tensor(tar_img2)

    pred_img2 = pred_img2.to(torch.int64)
    tar_img2 = tar_img2.to(torch.int64)

    metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    hd95 = metric(pred_img2, tar_img2)
    hd95 = torch.mean(hd95)

    print('name:',names)
    print('a2 hd95:',hd95)
                                          



