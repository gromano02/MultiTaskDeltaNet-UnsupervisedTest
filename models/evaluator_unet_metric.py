import numpy as np
import matplotlib.pyplot as plt
import os

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

from utils import de_norm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import utils

def CDEva():

    #### batch_size
    train_dataset,val_dataset,test_dataset = load_dataset()  ##### unet data
    train_pair, val_pair, test_pair = load_dataset_phy()     ##### pair data

    #test_pair_load = DataLoader(test_pair, batch_size=1, shuffle=False, num_workers=8)
    val_pair_load = DataLoader(val_pair, batch_size=1, shuffle=False, num_workers=8)

    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    dataloaders = {'train':trainloader, 'val':valloader, 'test':testloader}

    net_G = UNet(in_channels=3, out_channels=2, init_features=32)

    #PATH = '/home/yun13001/code/Carbon/tianyu_new_data/unet_github/checkpoints/unet_testa1_results/best_a1.pt'
    PATH = '/home/yun13001/code/Carbon/tianyu_new_data/unet_github/checkpoints/unet_testa2_results/best_a2.pt'

    running_metric1 = ConfuseMatrixMeter(n_class=4)
    running_metric2 = ConfuseMatrixMeter(n_class=4)


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
    """
    ##### multiple gpus load model
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            net_G = torch.nn.DataParallel(net_G)
    net_G.to(device)
    checkpoint = torch.load(PATH,weights_only=False)
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    epoch_save = checkpoint['epoch']
    net_G.eval()
    """

    ##### Unet prediction
    unet_pred=[]
    #for batch_id, batch in enumerate(dataloaders['test'], 0):
    for batch_id, batch in enumerate(dataloaders['val'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)
             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             unet_pred.append(pred)

    print('unet_pred',len(unet_pred))
    ##### make image pair based on differrent region
    test_label=[]
    ##### 103:46, 301:18     
    for m in range(46):

            numb =m

            for n in range(numb):

                ###### Label1 ######
                x_ref = unet_pred[m]
                x_res = unet_pred[n]

                #### overlap
                img = x_ref + x_res
                img_test=img
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                ####
                test_label.append(img_tt)

    for m in range(46,46+18):

            numb =m

            for n in range(46,numb):

                ###### Label1 ######
                x_ref = unet_pred[m]
                x_res = unet_pred[n]

                #### overlap
                img = x_ref + x_res
                img_test=img
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                ####
                test_label.append(img_tt)

    print('pred:',len(test_label))


    ##### Same order as the pair branch model:
    #for batch_id, batch in enumerate(test_pair_load, 0):
    for batch_id, batch in enumerate(val_pair_load, 0):
        with torch.no_grad():

            vis_input = utils.make_numpy_grid(de_norm(batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(batch['B']))

            gt1 = batch['L1'].to(device).long()
            gt2 = batch['L2'].to(device).long()

            gt1 = utils.make_numpy_grid_lb(gt1)
            gt2 = utils.make_numpy_grid_lb(gt2)

            #pred1 = test_label[batch_id]
            pred2 = test_label[batch_id]

            palette = np.array([[1, 1, 1],  # "no change"
                    [0.5, 0, 0],  # "appearing(1)"
                    [0, 0.5, 0],  # "disappearing(-1)"
                    [0.4,0.4,0.5]], dtype='float32')  # "overlap(2)"  [0.4,0.4,0.5]

            #vis_pred1 = palette[pred1.ravel()].reshape(vis_input.shape)
            #vis_gt1 = palette[gt1.ravel()].reshape(vis_input.shape)     ##### a1
            vis_pred2 = palette[pred2.ravel()].reshape(vis_input.shape)
            vis_gt2 = palette[gt2.ravel()].reshape(vis_input.shape)      ##### a2 

            #vis_dir = '/home/yun13001/code/Carbon/tianyu_new_data/unet_github/vis/unet/a1_save'
            vis_dir = '/home/yun13001/code/Carbon/tianyu_new_data/unet_github/vis/unet/a2_save'

            #vis = np.concatenate([vis_input, vis_input2, vis_gt1, vis_pred1], axis=0)
            vis = np.concatenate([vis_input, vis_input2, vis_gt2, vis_pred2], axis=0)

            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(vis_dir, 'eval_' + str(batch_id)+'.jpg')
            plt.imsave(file_name, vis)

        #target1 = batch['L1'].to(device).detach()
        #target1 = np.squeeze(target1)
        #G_pred1 = test_label[batch_id]
        #current_score1 = running_metric1.undate_score(pr=G_pred1, gt=target1.cpu().numpy())

        #message = '%s _A1: ,' % (batch_id)
        #for k, v in current_score1.items():
        #    message += '%s: %.5f ' % (k, v)
        #print(message)

        target2 = batch['L2'].to(device).detach()
        target2 = np.squeeze(target2)
        G_pred2 = test_label[batch_id]
        current_score2 = running_metric2.undate_score(pr=G_pred2, gt=target2.cpu().numpy())

        message = '%s _A2: ,' % (batch_id)
        for k, v in current_score2.items():
            message += '%s: %.5f ' % (k, v)
        print(message)


    #scores1 = running_metric1.get_scores()
    #val_acc1 = scores1['mf1']

    #print('test results:')
    #message = 'A1_'
    #for k, v in scores1.items():
    #    message += '%s: %.5f ' % (k, v)
    #print(message)


    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('test results:')
    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)    
                                                       
