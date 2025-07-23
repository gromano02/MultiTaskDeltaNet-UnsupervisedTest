
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from models.nets import *
from models.unet import UNet

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import get_alpha, get_alpha1, get_alpha2, softmax_helper, FocalLoss, mIoULoss, mmIoULoss, reg_term, reg_term_phy, cross_entropy
from models.loss_reg import reg_phy
from models.load_unet import load_dataset

##### When you need to run the prediction for a1 or a2, 
##### change thecorresponding line for a1 and a2. 

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"      ##### if train model on particulart GPU
torch.use_deterministic_algorithms(True)       ##### fix seed to reproduce results (delelte it if not needed)

def CDTrainer():
    
    ##### fix seed to reproduce results 
    seed = 8888
    torch.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed) 

    gen = torch.Generator()
    gen.manual_seed(seed)
    #####

    net_G = UNet(in_channels=3, out_channels=2, init_features=32)


    #optimizer_G = optim.AdamW(net_G.parameters(), lr=4.134219340349677e-05, betas=(0.9, 0.999), weight_decay=0.01) ### a1 hyper parameters
    optimizer_G = optim.AdamW(net_G.parameters(), lr=2.6174451128919035e-05, betas=(0.9, 0.999), weight_decay=0.01) ### a2 hyper parameters
    exp_lr_scheduler_G = get_scheduler(optimizer_G,500)  #####200

    running_metric = ConfuseMatrixMeter(n_class=2)     ###### 4 classes

    #### batch_size
    train_dataset,val_dataset,test_dataset = load_dataset()

    #trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)  ### a1 hyper
    #valloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=8)  
    ##trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)  ### a2 hyper
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8,worker_init_fn=seed_worker,generator=gen)  ##### fix seed 
    valloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=8)

    dataloaders = {'train':trainloader, 'val':valloader}
    print(len(dataloaders))


    #  training log
    epoch_acc = 0
    epoch_loss = 0     #### loss

    best_val_loss = 0 
    best_val_acc = 0.0

    best_epoch_id = 0
    best_epoch_id1 = 0
    best_epoch_id2 = 0
    epoch_to_start = 0
    max_num_epochs = 500
    #max_num_epochs = 15   ###for test

    global_step = 0
    steps_per_epoch = len(dataloaders['train'])
    total_steps = (max_num_epochs - epoch_to_start)*steps_per_epoch

    G_loss = None

    G_loss1 = None
    G_loss2 = None

    batch_id = 0
    epoch_id = 0

    ##### focal loss
    #alpha1           = get_alpha1(dataloaders['train']) # calculare class occurences
    #print(f"alpha-0 (no-change)={alpha1[0]}, alpha-1 (change)={alpha1[1]}")
    #_pxl_loss  = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha1, gamma = 2, smooth = 1e-5)
    alpha2           = get_alpha2(dataloaders['train']) # calculare class occurences
    print(f"alpha-0 (no-change)={alpha2[0]}, alpha-1 (change)={alpha2[1]}")
    _pxl_loss  = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha2, gamma = 2, smooth = 1e-5)


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"    #if only train on single GPU
        #device = "cuda" 
        if torch.cuda.device_count() > 1:
            net_G = nn.DataParallel(net_G)


    print(f"Using {device} device")
    net_G.to(device)

    print('training from scratch...')


    tri_loss=[]
    vali_loss=[]

    tri_acc=[]    ##### training f1 score for a1
    vali_acc=[]    ##### validation f1 score for a1

    for epoch_id in range(epoch_to_start, max_num_epochs):

        ################## train #################
        ##########################################
        train_loss=0.0
        train_loss1=0.0
        train_loss2=0.0
        train_reg = 0.0
        train_steps=0

        running_metric.clear()

        #is_training = True
        net_G.train()  # Set model to training mode

        for batch_id, batch in enumerate(dataloaders['train'], 0):
            #####Forward
            img_in = batch['A'].to(device)
            G_pred = net_G(img_in)        

            ######Optimize
            optimizer_G.zero_grad()

            #####Backward
            #gt = batch['L1'].to(device).long()
            gt = batch['L2'].to(device).long()

            G_loss = _pxl_loss(G_pred, gt)

            if epoch_id>=2000: #0(reg) ##### Epoch to apply regularization 
               ##### Add reg term
               y1  = batch['y1'].to(device)

            else:
                G_loss = G_loss   #### Loss for training
               
            G_loss.backward()
            optimizer_G.step()

            #target = batch['L1'].to(device).detach()
            target = batch['L2'].to(device).detach()
            G_pred = G_pred.detach()
            G_pred = torch.argmax(G_pred, dim=1)
            current_score = running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())

            train_loss += G_loss.item()
            train_steps += 1
            
        scores = running_metric.get_scores()
        train_acc = scores['mf1']

        exp_lr_scheduler_G.step() 

        print('train:')
        print('epoch',epoch_id)
        print(train_loss / train_steps)
        print(train_acc)

        tri_loss.append(train_loss / train_steps)
        tri_acc.append(train_acc)


        ################## Eval ##################
        ##########################################
        running_metric.clear()
        net_G.eval()


        val_loss=0
        val_loss=0
        val_steps=0

        # Iterate over data.
        for batch_id, batch in enumerate(dataloaders['val'], 0):
            with torch.no_grad():
                #####Forward
                img_in = batch['A'].to(device)
                G_pred = net_G(img_in)

                #####Backward
                #gt = batch['L1'].to(device).long()
                gt = batch['L2'].to(device).long()


                G_loss = _pxl_loss(G_pred, gt)

                #####Print statistics
                val_loss += G_loss.cpu().numpy()
                val_steps += 1

            #target = batch['L1'].to(device).detach()
            target = batch['L2'].to(device).detach()
            G_pred = G_pred.detach()
            G_pred = torch.argmax(G_pred, dim=1)
            current_score = running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())

        scores = running_metric.get_scores()
        val_acc = scores['mf1']

        print('validation:')
        print('epoch',epoch_id)
        print(val_loss / val_steps)
        print(val_acc)
        print('\n')

        vali_loss.append(val_loss / val_steps)
        vali_acc.append(val_acc)


        checkpoint_dir = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/unet/'    ##### save model
        #name='a1_results_test/'
        name='a2_results_test/'

        if not os.path.exists(checkpoint_dir+name):
            os.makedirs(checkpoint_dir+name)


        #### save best model (a1,a2)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch_id = epoch_id

            torch.save({'epoch':best_epoch_id,
                    'model_G_state_dict': net_G.state_dict(),
                    'optimizer_G_state_dict':optimizer_G.state_dict(),
                    'exp_lr_scheduler_G_state_dict': exp_lr_scheduler_G.state_dict(),
                    'val_acc': val_acc,
                    #},checkpoint_dir+name+'best_a1.pt')
                    },checkpoint_dir+name+'best_a2.pt')


        #### Save all the training and validation results
        np.save(checkpoint_dir+name+'train_loss.npy', tri_loss)
        np.save(checkpoint_dir+name+'train_acc.npy', tri_acc)

        np.save(checkpoint_dir+name+'val_loss.npy', vali_loss)
        np.save(checkpoint_dir+name+'val_acc.npy', vali_acc)



