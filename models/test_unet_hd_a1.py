import numpy as np
import matplotlib.pyplot as plt
import os
from monai.metrics import compute_hausdorff_distance, HausdorffDistanceMetric

from models.unet import UNet

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import get_alpha, get_alpha1, get_alpha2, softmax_helper, FocalLoss, mIoULoss, mmIoULoss, reg_term
from models.load_unet_phy import load_dataset_phy
from models.load_unet import load_dataset
from models.load_unet_s import load_dataset_s

from utils import de_norm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import utils

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

def CDTest(path):
    #### batch_size
    train_dataset,val_dataset,test_dataset = load_dataset()  ##### unet data
    train_pair, val_pair, test_pair = load_dataset_phy()     ##### pair data

    test_pair_load = DataLoader(test_pair, batch_size=1, shuffle=False, num_workers=8)
    val_pair_load = DataLoader(val_pair, batch_size=1, shuffle=False, num_workers=8)

    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    dataloaders = {'train':trainloader, 'val':valloader, 'test':testloader}

    net_G = UNet(in_channels=3, out_channels=2, init_features=32)

    PATH = path+'best_a1.pt'

    #### single gpu load model
    checkpoint = torch.load(PATH,weights_only=False)
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    epoch_save = checkpoint['epoch']
    #print('epoch_save',epoch_save)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G = torch.nn.DataParallel(net_G)
    net_G.to(device)
    net_G.eval()

    ##### Unet prediction:
    #### label (segmentation) prediction:
    #### test
    pred_img1=[]
    tar_img1 =[]
    for batch_id, batch in enumerate(dataloaders['test'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)
             img_gt = batch['L1'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             #pred = np.stack([pred, pred, pred], axis=-1)
             pred = pred.astype(np.int64)

             pred = binary_to_one_hot(pred) 
             pred = np.transpose(pred,(2,0,1))
             pred_img1.append(pred)

             target=img_gt.detach()
             target = np.squeeze(target)
             target = target.cpu() 

             target = binary_to_one_hot(target)
             target = np.transpose(target,(2,0,1))
             tar_img1.append(target)

    pred_img1= np.array(pred_img1)
    tar_img1 = np.array(tar_img1)

    pred_img1 = torch.tensor(pred_img1)
    tar_img1 = torch.tensor(tar_img1)

    pred_img1 = pred_img1.to(torch.int64)
    tar_img1 = tar_img1.to(torch.int64)

    metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    hd95 = metric(pred_img1, tar_img1)
    hd95 = torch.mean(hd95)

    print('test:')
    print('a1 hd95:',hd95)


    #### val
    pred_img1=[]
    tar_img1 =[]
    for batch_id, batch in enumerate(dataloaders['val'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)
             img_gt = batch['L1'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             #pred = np.stack([pred, pred, pred], axis=-1)
             pred = pred.astype(np.int64)

             pred = binary_to_one_hot(pred)
             pred = np.transpose(pred,(2,0,1))
             pred_img1.append(pred)

             target=img_gt.detach()
             target = np.squeeze(target)
             target = target.cpu()

             target = binary_to_one_hot(target)
             target = np.transpose(target,(2,0,1))
             tar_img1.append(target)

    pred_img1= np.array(pred_img1)
    tar_img1 = np.array(tar_img1)

    pred_img1 = torch.tensor(pred_img1)
    tar_img1 = torch.tensor(tar_img1)

    pred_img1 = pred_img1.to(torch.int64)
    tar_img1 = tar_img1.to(torch.int64)

    metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    hd95 = metric(pred_img1, tar_img1)
    hd95 = torch.mean(hd95)

    print('val:')
    print('a1 hd95:',hd95)


    ###### seperate results: 
    val_dataset1,val_dataset2,test_dataset1,test_dataset2 = load_dataset_s()

    valloader1 = DataLoader(val_dataset1, batch_size=1, shuffle=False, num_workers=8)
    testloader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=8)
    valloader2= DataLoader(val_dataset2, batch_size=1, shuffle=False, num_workers=8)
    testloader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=8)

    dataloaders = {'val1':valloader1, 'val2':valloader2, 'test1':testloader1, 'test2':testloader2}


    ##### Unet prediction:
    #### label prediction:
    #### 201
    pred_img1=[]
    tar_img1 =[]
    for batch_id, batch in enumerate(dataloaders['test1'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)
             img_gt = batch['L1'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             #pred = np.stack([pred, pred, pred], axis=-1)
             pred = pred.astype(np.int64)

             pred = binary_to_one_hot(pred)
             pred = np.transpose(pred,(2,0,1))
             pred_img1.append(pred)

             target=img_gt.detach()
             target = np.squeeze(target)
             target = target.cpu()

             target = binary_to_one_hot(target)
             target = np.transpose(target,(2,0,1))
             tar_img1.append(target)

    pred_img1= np.array(pred_img1)
    tar_img1 = np.array(tar_img1)

    pred_img1 = torch.tensor(pred_img1)
    tar_img1 = torch.tensor(tar_img1)

    pred_img1 = pred_img1.to(torch.int64)
    tar_img1 = tar_img1.to(torch.int64)

    metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    hd95 = metric(pred_img1, tar_img1)

    print('201_a1:',hd95)

    hd_img1 =hd95.tolist()
    hd_img1 = np.array(hd_img1)

    vis_dir1 = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/unet/a1'
    np.savetxt(vis_dir1+'/'+'a1_hd_'+'201'+".txt", hd_img1, fmt="%s", delimiter=",")

    hd95 = torch.mean(hd95)

    print('201 overall:')
    print('a1 hd95:',hd95)


    ##### Unet prediction:
    #### label prediction:
    #### 203
    pred_img1=[]
    tar_img1 =[]
    for batch_id, batch in enumerate(dataloaders['test2'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)
             img_gt = batch['L1'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             #pred = np.stack([pred, pred, pred], axis=-1)
             pred = pred.astype(np.int64)

             pred = binary_to_one_hot(pred)
             pred = np.transpose(pred,(2,0,1))
             pred_img1.append(pred)

             target=img_gt.detach()
             target = np.squeeze(target)
             target = target.cpu()

             target = binary_to_one_hot(target)
             target = np.transpose(target,(2,0,1))
             tar_img1.append(target)

    pred_img1= np.array(pred_img1)
    tar_img1 = np.array(tar_img1)

    pred_img1 = torch.tensor(pred_img1)
    tar_img1 = torch.tensor(tar_img1)

    pred_img1 = pred_img1.to(torch.int64)
    tar_img1 = tar_img1.to(torch.int64)

    metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    hd95 = metric(pred_img1, tar_img1)

    print('203_a1:',hd95)

    hd_img1 =hd95.tolist()
    hd_img1 = np.array(hd_img1)

    vis_dir1 = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/unet/a1'
    np.savetxt(vis_dir1+'/'+'a1_hd_'+'203'+".txt", hd_img1, fmt="%s", delimiter=",")

    hd95 = torch.mean(hd95)

    print('203 overall:')
    print('a1 hd95:',hd95)


    #### label prediction:
    #### 103
    pred_img1=[]
    tar_img1 =[]
    for batch_id, batch in enumerate(dataloaders['val1'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)
             img_gt = batch['L1'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             #pred = np.stack([pred, pred, pred], axis=-1)
             pred = pred.astype(np.int64)

             pred = binary_to_one_hot(pred)
             pred = np.transpose(pred,(2,0,1))
             pred_img1.append(pred)

             target=img_gt.detach()
             target = np.squeeze(target)
             target = target.cpu()

             target = binary_to_one_hot(target)
             target = np.transpose(target,(2,0,1))
             tar_img1.append(target)

    pred_img1= np.array(pred_img1)
    tar_img1 = np.array(tar_img1)

    pred_img1 = torch.tensor(pred_img1)
    tar_img1 = torch.tensor(tar_img1)

    pred_img1 = pred_img1.to(torch.int64)
    tar_img1 = tar_img1.to(torch.int64)

    metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    hd95 = metric(pred_img1, tar_img1)

    print('103_a1:',hd95)

    hd_img1 =hd95.tolist()
    hd_img1 = np.array(hd_img1)

    vis_dir1 = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/unet/a1'
    np.savetxt(vis_dir1+'/'+'a1_hd_'+'103'+".txt", hd_img1, fmt="%s", delimiter=",")

    hd95 = torch.mean(hd95)

    print('103 overall:')
    print('a1 hd95:',hd95)



    #### label prediction:
    #### 301
    pred_img1=[]
    tar_img1 =[]
    for batch_id, batch in enumerate(dataloaders['val2'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)
             img_gt = batch['L1'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             #pred = np.stack([pred, pred, pred], axis=-1)
             pred = pred.astype(np.int64)

             pred = binary_to_one_hot(pred)
             pred = np.transpose(pred,(2,0,1))
             pred_img1.append(pred)

             target=img_gt.detach()
             target = np.squeeze(target)
             target = target.cpu()

             target = binary_to_one_hot(target)
             target = np.transpose(target,(2,0,1))
             tar_img1.append(target)

    pred_img1= np.array(pred_img1)
    tar_img1 = np.array(tar_img1)

    pred_img1 = torch.tensor(pred_img1)
    tar_img1 = torch.tensor(tar_img1)

    pred_img1 = pred_img1.to(torch.int64)
    tar_img1 = tar_img1.to(torch.int64)

    metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    hd95 = metric(pred_img1, tar_img1)

    print('301_a1:',hd95)

    hd_img1 =hd95.tolist()
    hd_img1 = np.array(hd_img1)

    vis_dir1 = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/unet/a1'
    np.savetxt(vis_dir1+'/'+'a1_hd_'+'301'+".txt", hd_img1, fmt="%s", delimiter=",")

    hd95 = torch.mean(hd95)

    print('301 overall:')
    print('a1 hd95:',hd95)

