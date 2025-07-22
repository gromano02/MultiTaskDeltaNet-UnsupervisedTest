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
#from models.load_tianyu_cv import load_dataset_test
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

    running_metric1 = ConfuseMatrixMeter(n_class=4)
    running_metric2 = ConfuseMatrixMeter(n_class=4)

    PATH = '/home/yun13001/code/Carbon/tianyu_new_data/unet_github/checkpoints/unet_testa1_results/best_a1.pt'
    #PATH = '/home/yun13001/code/Carbon/tianyu_new_data/unet_github/checkpoints/unet_testa2_results/best_a2.pt'

    
    #### single gpu load model
    checkpoint = torch.load(PATH,weights_only=False)
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    epoch_save = checkpoint['epoch']

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

    ##### unet pred to image pair:
    test_label=[]
    for m in range(len(unet_pred)):

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
                ######  ######    
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

            pred1 = test_label[batch_id]
            #pred2 = test_label[batch_id]

            palette = np.array([[1, 1, 1],  # "no change"
                    [0.5, 0, 0],  # "appearing(1)"
                    [0, 0.5, 0],  # "disappearing(-1)"
                    [0.4,0.4,0.5]], dtype='float32')  # "overlap(2)"  [0.4,0.4,0.5]

            vis_pred1 = palette[pred1.ravel()].reshape(vis_input.shape)
            vis_gt1 = palette[gt1.ravel()].reshape(vis_input.shape)     ##### a1
            #vis_pred2 = palette[pred2.ravel()].reshape(vis_input.shape)
            #vis_gt2 = palette[gt2.ravel()].reshape(vis_input.shape)      ##### a2 

            vis_dir = '/home/yun13001/code/Carbon/tianyu_new_data/unet_github/vis/unet/a1_save'
            #vis_dir = '/home/yun13001/code/Carbon/tianyu_new_data/unet_github/vis/unet/a2_save'


            palette_app = np.array([[1, 1, 1],  # "no change"
                    [1.0, 0.6, 0.6],             # "appearing(1)"
                    [0.8, 1.0, 0.8],             # "disappearing(-1)"
                    [0.70,0.70,1.0],             # "overlap(2)"  [0.4,0.4,0.5]
                    [1.0, 0.0, 1.0],           # no change to appearing (Fuchsia) 
                    [0.2, 0.6, 0.4],           # no change to disappearing (339966)  
                    [0.28,0.28,0.42],          # no change to overlap (47476b) 
                    [0.7, 0, 0.35],         # appearing to no change (Rose red)
                    [1.0, 0.4, 0.0],        # appearing to disappearing (organge)
                    [0.4, 0.0, 0.4],        # appearing to overlap, 0.45,0.4,0.6 (purple)
                    [0.0, 0.9, 0.0],             # disappearing to no change
                    [0.6, 0.4, 0.0],             # disappearing to no appearing (996600)
                    [0.2, 0.2, 0.8],             # disappeaering to overlap (3333cc)
                    [0.5, 0.5, 0.5],        #### overlap to no change(808080)
                    [0.8, 0.4, 0.6],        #### overlap to appearing
                    [0.01, 0.5, 0.02],      #### overlap to disappearing(037d05)
                    ], dtype='float32')

            
            pred1_app = pred1.ravel().copy()
            #pred2_app = pred2.ravel().copy()
            gt1_app   = gt1.ravel().copy()
            gt2_app   = gt2.ravel().copy()

            
            for i in range(len(pred1_app)):
                if pred1_app[i]==0 and gt1_app[i]==1:
                    pred1_app[i]=4
                if pred1_app[i]==0 and gt1_app[i]==2:
                    pred1_app[i]=5
                if pred1_app[i]==0 and gt1_app[i]==3:
                    pred1_app[i]=6

                if pred1_app[i]==1 and gt1_app[i]==0:
                    pred1_app[i]=7
                if pred1_app[i]==1 and gt1_app[i]==2:
                    pred1_app[i]=8
                if pred1_app[i]==1 and gt1_app[i]==3:
                    pred1_app[i]=9

                if pred1_app[i]==2 and gt1_app[i]==0:
                    pred1_app[i]=10
                if pred1_app[i]==2 and gt1_app[i]==1:
                    pred1_app[i]=11
                if pred1_app[i]==2 and gt1_app[i]==3:
                    pred1_app[i]=12

                if pred1_app[i]==3 and gt1_app[i]==0:
                    pred1_app[i]=13
                if pred1_app[i]==3 and gt1_app[i]==1:
                    pred1_app[i]=14
                if pred1_app[i]==3 and gt1_app[i]==2:
                    pred1_app[i]=15
            """
            for i in range(len(pred2_app)):
                if pred2_app[i]==0 and gt2_app[i]==1:
                    pred2_app[i]=4
                if pred2_app[i]==0 and gt2_app[i]==2:
                    pred2_app[i]=5
                if pred2_app[i]==0 and gt2_app[i]==3:
                    pred2_app[i]=6

                if pred2_app[i]==1 and gt2_app[i]==0:
                    pred2_app[i]=7
                if pred2_app[i]==1 and gt2_app[i]==2:
                    pred2_app[i]=8
                if pred2_app[i]==1 and gt2_app[i]==3:
                    pred2_app[i]=9

                if pred2_app[i]==2 and gt2_app[i]==0:
                    pred2_app[i]=10
                if pred2_app[i]==2 and gt2_app[i]==1:
                    pred2_app[i]=11
                if pred2_app[i]==2 and gt2_app[i]==3:
                    pred2_app[i]=12

                if pred2_app[i]==3 and gt2_app[i]==0:
                    pred2_app[i]=13
                if pred2_app[i]==3 and gt2_app[i]==1:
                    pred2_app[i]=14
                if pred2_app[i]==3 and gt2_app[i]==2:
                    pred2_app[i]=15
            """

            vis_pred1_app = palette_app[pred1_app].reshape(vis_input.shape)
            #vis_pred2_app = palette_app[pred2_app].reshape(vis_input.shape)
            vis_gt1_app   = palette_app[gt1_app].reshape(vis_input.shape)
            vis_gt2_app   = palette_app[gt2_app].reshape(vis_input.shape)

            #vis_app = np.concatenate([vis_input, vis_input2, vis_gt1_app, vis_pred1_app, vis_gt2_app, vis_pred2_app], axis=0)
            vis_app = np.concatenate([vis_input, vis_input2, vis_gt1_app, vis_pred1_app], axis=0)
            #vis_app = np.concatenate([vis_input, vis_input2, vis_gt2_app, vis_pred2_app], axis=0)
            vis_app = np.clip(vis_app, a_min=0.0, a_max=1.0)

            file_name = os.path.join(vis_dir, 'con_max1_' + str(batch_id)+'.png')
            #file_name = os.path.join(vis_dir, 'con_max2_' + str(batch_id)+'.png')

            plt.imsave(file_name, vis_app)
            plt.close()

            #### confusion matrix
            class_names = ['no_change','appearing','disappearing','overlap']

            cm1 = confusion_matrix(gt1.ravel(), pred1.ravel())
            #cm2 = confusion_matrix(gt2.ravel(), pred2.ravel())

            if cm1.shape[0]!=4 or cm1.shape[1]!=4:
                continue
            #if cm2.shape[0]!=4 or cm2.shape[1]!=4:
            #    continue

            fig = plt.figure(figsize=(30, 26))
            ax= plt.subplot()
            sns.set(font_scale=3)
            colormap = sns.color_palette("Blues") 
            sns.heatmap(cm1, annot=True, ax = ax, fmt = 'g', cmap=colormap, annot_kws={'size': 30}); #annot=True to annotate cells
            #sns.heatmap(cm2, annot=True, ax = ax, fmt = 'g', cmap=colormap, annot_kws={'size': 30}); #annot=True to annotate cells
            # labels, title and ticks
            ax.set_xlabel('Predicted', fontsize=35)
            ax.xaxis.set_label_position('bottom')
            #plt.xticks(rotation=90)
            ax.xaxis.set_ticklabels(class_names, fontsize = 30)
            ax.xaxis.tick_bottom()

            ax.set_ylabel('True', fontsize=35)
            ax.yaxis.set_ticklabels(class_names, fontsize = 30)
            plt.yticks(rotation=0)

            plt.title('Confusion Matrix a1', fontsize=35)
            plt.savefig(vis_dir+'/'+'ConMat1_'+str(batch_id)+'.png')
            #plt.title('Confusion Matrix a2', fontsize=35)
            #plt.savefig(vis_dir+'/'+'ConMat2_'+str(batch_id)+'.png')            
            plt.close()


