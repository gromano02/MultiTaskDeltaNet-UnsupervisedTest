import numpy as np
import matplotlib.pyplot as plt
import os
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.test_siam_cd3 import *
#from models.test_siam_cd2 import *
#from models.test_siam_cd import *

from models.nets import *
from models.sia_two_out_four_class import Siamese
#from models.siamese_with_weights import Siamese

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import get_alpha, get_alpha1, get_alpha2, softmax_helper, FocalLoss, mIoULoss, mmIoULoss, reg_term, reg_term_phy
from models.losses_reg import reg_phy
#from models.load_cv import load_dataset
from models.load_cv2 import load_dataset
#from models.load_phy import load_dataset

torch.use_deterministic_algorithms(True)

def CDTrainer():

    seed = 8888               ### 88, 8, 888
    torch.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    gen = torch.Generator()
    gen.manual_seed(seed)


    #### init_fratures ()
    net_G = Siamese(in_channels=3, out_channels=4, init_features=32)

    #### learning_rate, weight_decay()
    #### cv_all_pair
    optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.0009551981358212413, betas=(0.9, 0.999), weight_decay=0.01)  ### test1
    #optimizer_G = optim.AdamW(net_G.parameters(), lr=0.00037037312300132946, betas=(0.9, 0.999), weight_decay=0.01)  ###cv_all_pair(workstation3) test2
    #### unet4:                                                                                                             
    #optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.00041403070497565746, betas=(0.9, 0.999), weight_decay=0.01)    ####0.00022013816468482175
    #### unet3:
    #optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.0004992201208100422, betas=(0.9, 0.999), weight_decay=0.01)  ####0.0004992201208100422, 0.00041836579466918123
    #optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.00041836579466918123, betas=(0.9, 0.999), weight_decay=0.01) 
    #### unet2:
    #optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.0005898338217031561, betas=(0.9, 0.999), weight_decay=0.01)
    #optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.0015543985658156383, betas=(0.9, 0.999), weight_decay=0.01)
    #optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.0005460622879178106, betas=(0.9, 0.999), weight_decay=0.01)
    #optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.0001950109194261348, betas=(0.9, 0.999), weight_decay=0.01)
    #### unet1:
    #optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.00048497886264844617, betas=(0.9, 0.999), weight_decay=0.01)
    #optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.0002544826874211984, betas=(0.9, 0.999), weight_decay=0.01)
    #optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.0006255805193004037, betas=(0.9, 0.999), weight_decay=0.01)
    #optimizer_G = optim.AdamW(net_G.parameters(), lr= 0.0004939894048343724, betas=(0.9, 0.999), weight_decay=0.01)
    exp_lr_scheduler_G = get_scheduler(optimizer_G,500) ###200


    ### train metric
    running_metric1 = ConfuseMatrixMeter(n_class=4)
    running_metric2 = ConfuseMatrixMeter(n_class=4)

    ### val metric
    running_metric1_1 = ConfuseMatrixMeter(n_class=4)    #a1 #103  ###### 4 classes
    running_metric1_2 = ConfuseMatrixMeter(n_class=4)    #301
    running_metric2_1 = ConfuseMatrixMeter(n_class=4)    #a2 #103
    running_metric2_2 = ConfuseMatrixMeter(n_class=4)    #301

    #### batch_size
    train_dataset,val_dataset1,val_dataset2,test_dataset = load_dataset()

    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8,worker_init_fn=seed_worker,generator=gen) ##### cv best
    valloader1 = DataLoader(val_dataset1, batch_size=16, shuffle=True, num_workers=8)     ##### cv best
    valloader2 = DataLoader(val_dataset2, batch_size=16, shuffle=True, num_workers=8)

    #trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, worker_init_fn=seed_worker, generator=gen)
    #valloader1 = DataLoader(val_dataset1, batch_size=8, shuffle=True, num_workers=8)
    #valloader2 = DataLoader(val_dataset2, batch_size=8, shuffle=True, num_workers=8)    

    dataloaders = {'train':trainloader, 'val1':valloader1, 'val2':valloader2}
    print(len(dataloaders))

    #  training log
    epoch_acc = 0
    epoch_loss = 0     #### loss

    best_val_loss = 0  ####
    best_val_acc1 = 0.0
    best_val_acc2 = 0.0
    best_val_acc_m= 0.0

    best_epoch_id = 0
    best_epoch_id1 = 0
    best_epoch_id2 = 0
    epoch_to_start = 0
    max_num_epochs = 500
    #max_num_epochs = 100
    #max_num_epochs = 1

    early_stop=50   #100
    improve_epoch=0 ###early stop

    global_step = 0
    steps_per_epoch = len(dataloaders['train'])
    total_steps = (max_num_epochs - epoch_to_start)*steps_per_epoch

    G_pred1 = None
    G_pred2 = None
    batch = None
    G_loss = None
    G_loss_ = None   #### to save loss value without reg

    G_loss1 = None
    G_loss2 = None

    batch_id = 0
    epoch_id = 0


    ##### focal loss
    alpha1           = get_alpha1(dataloaders['train']) # calculare class occurences
    print(f"alpha-0 (no-change)={alpha1[0]}, alpha-1 (appearance)={alpha1[1]}, alpha-2 (disappearance)={alpha1[2]}, alpha-3 (overlap)={alpha1[3]}")
    _pxl_loss1  = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha1, gamma = 2, smooth = 1e-5)

    alpha2           = get_alpha2(dataloaders['train']) # calculare class occurences
    print(f"alpha-0 (no-change)={alpha2[0]}, alpha-1 (appearance)={alpha2[1]}, alpha-2 (disappearance)={alpha2[2]}, alpha-3 (overlap)={alpha2[3]}")
    _pxl_loss2  = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha2, gamma = 2, smooth = 1e-5)


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G = nn.DataParallel(net_G)
    net_G.to(device)

    print('training from scratch...')

    tri_loss=[]
    tri_loss1=[]
    tri_loss2=[]

    tri_acc1=[]    ##### training f1 score for a1
    tri_acc2=[]

    tri_reg=[]     ##### training regularization value

    vali_loss=[]
    vali_loss1=[]
    vali_loss2=[]

    vali_acc1=[]    ##### validation f1 score for a1
    vali_acc2=[]


    for epoch_id in range(epoch_to_start, max_num_epochs):

        ################## train #################
        ##########################################
        train_loss=0.0
        train_loss1=0.0
        train_loss2=0.0
        train_reg = 0.0
        train_steps=0

        running_metric1.clear()
        running_metric2.clear()

        net_G.train()  # Set model to training mode

        print('begin training:')
        for batch_id, batch in enumerate(dataloaders['train'], 0):
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G(img_in1, img_in2)

            #####Optimize
            optimizer_G.zero_grad()

            #####Backward
            gt1 = batch['L1'].to(device).long()
            gt2 = batch['L2'].to(device).long()

            G_loss1 = _pxl_loss1(G_pred1, gt1)
            G_loss2 = _pxl_loss2(G_pred2, gt2)

            G_loss = G_loss1 + G_loss2
            ####
            if epoch_id>=2000: #0(reg)  #1000(no reg)   ##### Epoch to apply regularization
               ##### Reg term
               y1  = batch['y1'].to(device)
               y2  = batch['y2'].to(device)
               Len = batch['Len'].to(device)
               Pi = batch['Pi'].to(device)
               max_ = batch['max'].to(device)
               min_ = batch['min'].to(device)

               reg = reg_phy(gt1,gt2,G_pred1,G_pred2,y1,y2,max_,min_,Len,Pi)

               G_loss_ = G_loss  #### Save loss without reg term

               #### Loss weight
               lam =2 ###
               #G_loss = G_loss + reg  #### Loss for training
               G_loss = G_loss + lam*reg  #### Loss for training

               train_reg = train_reg+reg.item()
               #print('reg training')
            else:
               G_loss_ = G_loss  #### Save loss without reg term
               G_loss = G_loss   #### Loss for training


            G_loss.backward()
            optimizer_G.step()

            target1 = batch['L1'].to(device).detach()
            G_pred1 = G_pred1.detach()
            G_pred1 = torch.argmax(G_pred1, dim=1)
            current_score1 = running_metric1.update_cm(pr=G_pred1.cpu().numpy(), gt=target1.cpu().numpy())

            target2 = batch['L2'].to(device).detach()
            G_pred2 = G_pred2.detach()
            G_pred2 = torch.argmax(G_pred2, dim=1)
            current_score2 = running_metric2.update_cm(pr=G_pred2.cpu().numpy(), gt=target2.cpu().numpy())

            #####Print statistics
            train_loss1 += G_loss1.item()
            train_loss2 += G_loss2.item()

            #train_loss += G_loss.item()
            train_loss += G_loss_.item()
            train_steps += 1


        scores1 = running_metric1.get_scores()
        train_acc1 = scores1['mf1']

        scores2 = running_metric2.get_scores()
        train_acc2 = scores2['mf1']

        exp_lr_scheduler_G.step()

        print('train:')
        print('epoch',epoch_id)
        print(train_loss / train_steps)
        print(train_acc1)
        print(train_acc2)

        tri_loss.append(train_loss / train_steps)
        tri_loss1.append(train_loss1 / train_steps)
        tri_loss2.append(train_loss2 / train_steps)

        tri_acc1.append(train_acc1)
        tri_acc2.append(train_acc2)

        tri_reg.append(train_reg / train_steps)


        ################## Eval ##################
        ##########################################
        running_metric1_1.clear()
        running_metric1_2.clear()
        running_metric2_1.clear()
        running_metric2_2.clear()

        net_G.eval()

        val_loss=0
        val_loss1=0
        val_loss2=0
        val_steps=0

        # Iterate over data.
        for batch_id, batch in enumerate(dataloaders['val1'], 0):
            with torch.no_grad():
                #####Forward
                img_in1 = batch['A'].to(device)
                img_in2 = batch['B'].to(device)
                G_pred1, G_pred2 = net_G(img_in1, img_in2)

                #####Backward
                gt1 = batch['L1'].to(device).long()
                gt2 = batch['L2'].to(device).long()

                ##### Reg term
                #reg = reg_term(gt1,gt2,G_pred1,G_pred2)

                G_loss1 = _pxl_loss1(G_pred1, gt1)
                G_loss2 = _pxl_loss2(G_pred2, gt2)
                G_loss = G_loss1 + G_loss2
                ####

                #####Print statistics
                val_loss1 += G_loss1.cpu().numpy()
                val_loss2 += G_loss2.cpu().numpy()

                val_loss += G_loss.cpu().numpy()
                val_steps += 1

            target1 = batch['L1'].to(device).detach()
            G_pred1 = G_pred1.detach()
            G_pred1 = torch.argmax(G_pred1, dim=1)
            current_score1_1 = running_metric1_1.update_cm(pr=G_pred1.cpu().numpy(), gt=target1.cpu().numpy())

            target2 = batch['L2'].to(device).detach()
            G_pred2 = G_pred2.detach()
            G_pred2 = torch.argmax(G_pred2, dim=1)
            current_score2_1 = running_metric2_1.update_cm(pr=G_pred2.cpu().numpy(), gt=target2.cpu().numpy())

        scores1_1 = running_metric1_1.get_scores()
        val_acc1_1 = scores1_1['mf1']

        scores2_1 = running_metric2_1.get_scores()
        val_acc2_1 = scores2_1['mf1']
        # Iterate over data.
        for batch_id, batch in enumerate(dataloaders['val2'], 0):
            with torch.no_grad():
                #####Forward
                img_in1 = batch['A'].to(device)
                img_in2 = batch['B'].to(device)
                G_pred1, G_pred2 = net_G(img_in1, img_in2)

                #####Backward
                gt1 = batch['L1'].to(device).long()
                gt2 = batch['L2'].to(device).long()

                ##### Reg term
                #reg = reg_term(gt1,gt2,G_pred1,G_pred2)

                G_loss1 = _pxl_loss1(G_pred1, gt1)
                G_loss2 = _pxl_loss2(G_pred2, gt2)
                G_loss = G_loss1 + G_loss2
                ####
                #G_loss = G_loss + reg                

                #####Print statistics
                val_loss1 += G_loss1.cpu().numpy()
                val_loss2 += G_loss2.cpu().numpy()

                val_loss += G_loss.cpu().numpy()
                val_steps += 1

            target1 = batch['L1'].to(device).detach()
            G_pred1 = G_pred1.detach()
            G_pred1 = torch.argmax(G_pred1, dim=1)
            current_score1_2 = running_metric1_2.update_cm(pr=G_pred1.cpu().numpy(), gt=target1.cpu().numpy())

            target2 = batch['L2'].to(device).detach()
            G_pred2 = G_pred2.detach()
            G_pred2 = torch.argmax(G_pred2, dim=1)
            current_score2_2 = running_metric2_2.update_cm(pr=G_pred2.cpu().numpy(), gt=target2.cpu().numpy())

        scores1_2 = running_metric1_2.get_scores()
        val_acc1_2 = scores1_1['mf1']

        scores2_2 = running_metric2_2.get_scores()
        val_acc2_2 = scores2_1['mf1']

        #### average f1 score 
        val_acc1 = (val_acc1_1+val_acc1_2)/2
        val_acc2 = (val_acc2_1+val_acc2_2)/2


        #mean_acc = (val_acc1+val_acc2)/2
        val_acc_m = (val_acc1+val_acc2)/2

        print('validation:')
        print('epoch',epoch_id)
        print(val_loss / val_steps)
        print(val_acc1)
        print(val_acc2)
        print('\n')

        vali_loss.append(val_loss / val_steps)
        vali_loss1.append(val_loss1 / val_steps)
        vali_loss2.append(val_loss2 / val_steps)

        vali_acc1.append(val_acc1)
        vali_acc2.append(val_acc2)

        checkpoint_dir = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/ablation/save_model/'

        name='save_model/'

        if not os.path.exists(checkpoint_dir+name):
            os.makedirs(checkpoint_dir+name)

        path = 'model.pt'

        #### Save the last epoch model
        torch.save({'epoch':epoch_id,
                    'model_G_state_dict': net_G.state_dict(),
                    'optimizer_G_state_dict':optimizer_G.state_dict(),
                    'exp_lr_scheduler_G_state_dict': exp_lr_scheduler_G.state_dict(),
                    'val_acc1': val_acc1,
                    'val_acc2': val_acc2,
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
                    'val_acc2': val_acc2,
                    },checkpoint_dir+name+'best_a1.pt')


        if val_acc2 > best_val_acc2:
            best_val_acc2 = val_acc2
            best_epoch_id2 = epoch_id
            print('save a2 model at:',best_epoch_id2)

            improve_epoch=0

            torch.save({'epoch':best_epoch_id2,
                    'model_G_state_dict': net_G.state_dict(),
                    'optimizer_G_state_dict':optimizer_G.state_dict(),
                    'exp_lr_scheduler_G_state_dict': exp_lr_scheduler_G.state_dict(),
                    'val_acc1': val_acc1,
                    'val_acc2': val_acc2,
                    },checkpoint_dir+name+'best_a2.pt')


        if val_acc_m > best_val_acc_m:
            best_val_acc_m = val_acc_m
            best_epoch_id_m = epoch_id
            print('save a_m model at:',best_epoch_id_m)

            improve_epoch=0

            torch.save({'epoch':best_epoch_id_m,
                    'model_G_state_dict': net_G.state_dict(),
                    'optimizer_G_state_dict':optimizer_G.state_dict(),
                    'exp_lr_scheduler_G_state_dict': exp_lr_scheduler_G.state_dict(),
                    'val_acc1': val_acc1,
                    'val_acc2': val_acc2,
                    },checkpoint_dir+name+'best_m.pt')


        #### Save all the training and validation results
        np.save(checkpoint_dir+name+'train_loss.npy', tri_loss)
        np.save(checkpoint_dir+name+'train_loss1.npy', tri_loss1)
        np.save(checkpoint_dir+name+'train_loss2.npy', tri_loss2)

        np.save(checkpoint_dir+name+'train_reg.npy', tri_reg)

        np.save(checkpoint_dir+name+'train_acc1.npy', tri_acc1)
        np.save(checkpoint_dir+name+'train_acc2.npy', tri_acc2)

        np.save(checkpoint_dir+name+'val_loss.npy', vali_loss)
        np.save(checkpoint_dir+name+'val_loss1.npy', vali_loss1)
        np.save(checkpoint_dir+name+'val_loss2.npy', vali_loss2)

        np.save(checkpoint_dir+name+'val_acc1.npy', vali_acc1)
        np.save(checkpoint_dir+name+'val_acc2.npy', vali_acc2)

    print('model (a1,a2) saved at',best_epoch_id1,best_epoch_id2)
    path = checkpoint_dir+name

    CDTest(path)

