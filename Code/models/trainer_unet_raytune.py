import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile

from models.nets import *
from models.unet import UNet

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import get_alpha, get_alpha1, get_alpha2, softmax_helper, FocalLoss, mIoULoss, mmIoULoss
from models.load_unet import load_dataset

import ray
from ray import tune,train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler


def CDTrainer(config):

    seed = 8888
    torch.manual_seed(seed)

    net_G = UNet(in_channels=3, out_channels=2, init_features=32)

    optimizer_G = optim.AdamW(net_G.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    exp_lr_scheduler_G = get_scheduler(optimizer_G,500)  #####200

    running_metric = ConfuseMatrixMeter(n_class=2)     ###### 4 classes


    #### batch_size
    train_dataset,val_dataset,test_dataset = load_dataset()

    trainloader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8)
    valloader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8)

    dataloaders = {'train':trainloader, 'val':valloader}
    print(len(dataloaders))



    #  training log
    epoch_acc = 0
    epoch_loss = 0     #### loss

    best_val_loss = 0  ####
    best_val_acc = 0.0

    best_epoch_id = 0
    best_epoch_id1 = 0
    best_epoch_id2 = 0
    epoch_to_start = 0
    max_num_epochs = 500 #200
    #max_num_epochs = 4

    global_step = 0
    steps_per_epoch = len(dataloaders['train'])
    total_steps = (max_num_epochs - epoch_to_start)*steps_per_epoch

    G_pred1 = None
    G_pred2 = None
    pred_vis = None
    batch = None
    G_loss = None
    G_loss_ = None   #### to save loss value without reg

    G_loss1 = None
    G_loss2 = None

    is_training = False
    batch_id = 0
    epoch_id = 0


    ##### focal loss
    #alpha1           = get_alpha1(dataloaders['train']) # calculare class occurences
    #print('alpha shape:',len(alpha1))
    #print(f"alpha-0 (no-change)={alpha1[0]}, alpha-1 (change)={alpha1[1]}")
    alpha2           = get_alpha2(dataloaders['train']) # calculare class occurences
    print(f"alpha-0 (no-change)={alpha2[0]}, alpha-1 (change)={alpha2[1]}")
    #_pxl_loss  = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha1, gamma = 2, smooth = 1e-5)
    _pxl_loss  = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha2, gamma = 2, smooth = 1e-5)


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G = nn.DataParallel(net_G)
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

        is_training = True
        net_G.train()  # Set model to training mode


        for batch_id, batch in enumerate(dataloaders['train'], 0):
            #####Forward
            img_in = batch['A'].to(device)
            G_pred = net_G(img_in)
            #print('G_pred.shape',G_pred.shape)

            #####Optimize
            optimizer_G.zero_grad()

            #####Backward
            #gt = batch['L1'].to(device).long()
            gt = batch['L2'].to(device).long()
            #print('gt.shape',gt.shape)

            G_loss = _pxl_loss(G_pred, gt)


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

        print('train:')
        print('epoch',epoch_id)
        print(train_loss / train_steps)
        print(train_acc)



        ################## Eval ##################
        ##########################################
        running_metric.clear()
        is_training = False
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

        print('test:')
        print('epoch',epoch_id)
        print(val_loss / val_steps)
        print(val_acc)
        print('\n')


        checkpoint_data = {
            "epoch": epoch_id,
            "net_state_dict": net_G.state_dict(),
            "optimizer_state_dict": optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': exp_lr_scheduler_G.state_dict(),
        }


        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            #checkpoint = None
            #if (epoch_id + 1) % 5 == 0:      ##### if more trails be saved during searching
            #     #This saves the model to the trial directory
            #    torch.save(
            #        net_G.state_dict(),
            #        os.path.join(temp_checkpoint_dir, "model.pth")
            #    )
            #    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Send the current training result back to Tune
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report( {"loss": val_loss / val_steps, "f1": val_acc}, checkpoint=checkpoint)        

        


