import numpy as np
import matplotlib.pyplot as plt
import os
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.test_siam_ablation import *

from models.nets import *
from models.sia_one_out import Siamese 

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import get_alpha, get_alpha1, get_alpha2, softmax_helper, FocalLoss, mIoULoss, mmIoULoss, reg_term, reg_term_phy
from models.load_cv2 import load_dataset

##### When you need to run the prediction for a1 or a2,
##### change the corresponding line for a1 and a2.

os.environ["CUDA_VISIBLE_DEVICES"] = "0"               ##### if train model on particulart GPU
torch.use_deterministic_algorithms(True)               ##### fix seed to reproduce results (delelte it if not needed)

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

    net_G = Siamese(in_channels=3, out_channels=4, init_features=32)

    #### learning_rate, weight_decay()
    #### no_init:
    optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.0009551981358212413, betas=(0.9, 0.999), weight_decay=0.01)

    exp_lr_scheduler_G = get_scheduler(optimizer_G,500)  


    ### train metric
    running_metric1 = ConfuseMatrixMeter(n_class=4)
    ### val metric
    running_metric1_1 = ConfuseMatrixMeter(n_class=4)    #a1 #103  ###### 4 classes
    running_metric1_2 = ConfuseMatrixMeter(n_class=4)    #301


    #### batch_size
    train_dataset,val_dataset1,val_dataset2,test_dataset = load_dataset()

    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8,worker_init_fn=seed_worker,generator=gen) 
    valloader1 = DataLoader(val_dataset1, batch_size=16, shuffle=True, num_workers=8)    
    valloader2 = DataLoader(val_dataset2, batch_size=16, shuffle=True, num_workers=8)

    dataloaders = {'train':trainloader, 'val1':valloader1, 'val2':valloader2}
    print(len(dataloaders))


    #  training log
    epoch_acc = 0
    epoch_loss = 0     #### loss

    best_val_loss = 0  ####
    best_val_acc1 = 0.0

    best_epoch_id = 0
    best_epoch_id1 = 0
    epoch_to_start = 0
    max_num_epochs = 500 

    early_stop=50   #100
    improve_epoch=0 ###early stop

    global_step = 0
    steps_per_epoch = len(dataloaders['train'])
    total_steps = (max_num_epochs - epoch_to_start)*steps_per_epoch

    G_pred1 = None
    G_pred2 = None
    batch = None
    G_loss = None

    G_loss1 = None

    batch_id = 0
    epoch_id = 0

    ##### focal loss
    #alpha1           = get_alpha1(dataloaders['train']) # calculare class occurences
    #print(f"alpha-0 (no-change)={alpha1[0]}, alpha-1 (appearance)={alpha1[1]}, alpha-2 (disappearance)={alpha1[2]}, alpha-3 (overlap)={alpha1[3]}")
    #_pxl_loss1  = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha1, gamma = 2, smooth = 1e-5)

    alpha2           = get_alpha2(dataloaders['train']) # calculare class occurences
    print(f"alpha-0 (no-change)={alpha2[0]}, alpha-1 (appearance)={alpha2[1]}, alpha-2 (disappearance)={alpha2[2]}, alpha-3 (overlap)={alpha2[3]}")
    _pxl_loss1  = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha2, gamma = 2, smooth = 1e-5)


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G = nn.DataParallel(net_G)
    net_G.to(device)

    print('training from scratch...')


    tri_loss=[]
    tri_loss1=[]

    tri_acc1=[]    ##### training f1 score for a1

    vali_loss=[]
    vali_loss1=[]

    vali_acc1=[]    ##### validation f1 score for a1


    for epoch_id in range(epoch_to_start, max_num_epochs):

        ################## train #################
        ##########################################
        train_loss=0.0
        train_loss1=0.0
        train_reg = 0.0
        train_steps=0

        running_metric1.clear()
        net_G.train()  # Set model to training mode

        print('begin training:')
        for batch_id, batch in enumerate(dataloaders['train'], 0):
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1 = net_G(img_in1, img_in2)

            #####Optimize
            optimizer_G.zero_grad()

            #####Backward
            #gt1 = batch['L1'].to(device).long()
            gt1 = batch['L2'].to(device).long()

            G_loss1 = _pxl_loss1(G_pred1, gt1)
            G_loss = G_loss1

            G_loss.backward()
            optimizer_G.step()

            #target1 = batch['L1'].to(device).detach()
            target1 = batch['L2'].to(device).detach()
            G_pred1 = G_pred1.detach()
            G_pred1 = torch.argmax(G_pred1, dim=1)
            current_score1 = running_metric1.update_cm(pr=G_pred1.cpu().numpy(), gt=target1.cpu().numpy())

            #####Print statistics
            train_loss += G_loss.item()
            train_steps += 1

        scores1 = running_metric1.get_scores()
        train_acc1 = scores1['mf1']

        exp_lr_scheduler_G.step()

        print('train:')
        print('epoch',epoch_id)
        print(train_loss / train_steps)
        print(train_acc1)

        tri_loss.append(train_loss / train_steps)


        ################## Eval ##################
        ##########################################
        running_metric1_1.clear()
        running_metric1_2.clear()

        net_G.eval()

        val_loss=0
        val_loss1=0
        val_steps=0

        # Iterate over data.
        for batch_id, batch in enumerate(dataloaders['val1'], 0):
            with torch.no_grad():
                #####Forward
                img_in1 = batch['A'].to(device)
                img_in2 = batch['B'].to(device)
                G_pred1 = net_G(img_in1, img_in2)

                #####Backward
                #gt1 = batch['L1'].to(device).long()
                gt1 = batch['L2'].to(device).long()

                G_loss1 = _pxl_loss1(G_pred1, gt1)
                G_loss = G_loss1

                #####Print statistics
                val_loss += G_loss1.cpu().numpy()
                val_steps += 1

            #target1 = batch['L1'].to(device).detach()
            target1 = batch['L2'].to(device).detach()
            G_pred1 = G_pred1.detach()
            G_pred1 = torch.argmax(G_pred1, dim=1)
            current_score1_1 = running_metric1_1.update_cm(pr=G_pred1.cpu().numpy(), gt=target1.cpu().numpy())

        scores1_1 = running_metric1_1.get_scores()
        val_acc1_1 = scores1_1['mf1']


        # Iterate over data.
        for batch_id, batch in enumerate(dataloaders['val2'], 0):
            with torch.no_grad():
                #####Forward
                img_in1 = batch['A'].to(device)
                img_in2 = batch['B'].to(device)
                G_pred1 = net_G(img_in1, img_in2)

                #####Backward
                #gt1 = batch['L1'].to(device).long()
                gt1 = batch['L2'].to(device).long()

                G_loss1 = _pxl_loss1(G_pred1, gt1)
                G_loss = G_loss1

                #####Print statistics
                val_loss += G_loss.cpu().numpy()
                val_steps += 1

            #target1 = batch['L1'].to(device).detach()
            target1 = batch['L2'].to(device).detach()
            G_pred1 = G_pred1.detach()
            G_pred1 = torch.argmax(G_pred1, dim=1)
            current_score1_2 = running_metric1_2.update_cm(pr=G_pred1.cpu().numpy(), gt=target1.cpu().numpy())

        scores1_2 = running_metric1_2.get_scores()
        val_acc1_2 = scores1_1['mf1']


        #### average f1 score 
        val_acc1 = (val_acc1_1+val_acc1_2)/2
        val_acc_m = val_acc1

        print('validation:')
        print('epoch',epoch_id)
        print(val_loss / val_steps)
        print(val_acc1)
        print('\n')

        vali_loss.append(val_loss / val_steps)
        vali_loss1.append(val_loss1 / val_steps)

        vali_acc1.append(val_acc1)

        checkpoint_dir = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/ablation/'     ###### save model
        #name='out1_test/'
        name='out2_test/'

        if not os.path.exists(checkpoint_dir+name):
            os.makedirs(checkpoint_dir+name)

        path = 'model.pt'

        #### Save the last epoch model
        torch.save({'epoch':epoch_id,
                    'model_G_state_dict': net_G.state_dict(),
                    'optimizer_G_state_dict':optimizer_G.state_dict(),
                    'exp_lr_scheduler_G_state_dict': exp_lr_scheduler_G.state_dict(),
                    'val_acc1': val_acc1,
                    #'val_acc2': val_acc2,
                    },checkpoint_dir+name+path)

        #### early stop 
        improve_epoch = improve_epoch+1
        if improve_epoch == early_stop:
            print('early stop:',epoch_id)
            break


        #### save best model (a1,a2)
        if val_acc1 > best_val_acc1:
            best_val_acc1 = val_acc1
            best_epoch_id1 = epoch_id
            print('save a1 model at:',best_epoch_id1)

            improve_epoch=0

            torch.save({'epoch':best_epoch_id1,
                    'model_G_state_dict': net_G.state_dict(),
                    'optimizer_G_state_dict':optimizer_G.state_dict(),
                    'exp_lr_scheduler_G_state_dict': exp_lr_scheduler_G.state_dict(),
                    'val_acc1': val_acc1,
                    #'val_acc2': val_acc2,
                    #},checkpoint_dir+name+'best_a1.pt')
                    },checkpoint_dir+name+'best_a2.pt')    

        #### Save all the training and validation results
        np.save(checkpoint_dir+name+'train_loss.npy', tri_loss)
        np.save(checkpoint_dir+name+'train_acc1.npy', tri_acc1)

        np.save(checkpoint_dir+name+'val_loss.npy', vali_loss)
        np.save(checkpoint_dir+name+'val_acc1.npy', vali_acc1)

    #print('model (a1) saved at',best_epoch_id1)
    print('model (a2) saved at',best_epoch_id1)
    path = checkpoint_dir+name

    CDTest(path)



