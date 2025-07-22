import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile

from models.nets import *
from models.sia_two_out_four_class import Siamese

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import get_alpha, get_alpha1, get_alpha2, softmax_helper, FocalLoss, mIoULoss, mmIoULoss, reg_term, reg_term_phy
from models.loss_reg import reg_phy
#from models.load_cv import load_dataset
from models.load_phy import load_dataset

import ray
from ray import tune,train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

torch.use_deterministic_algorithms(True)

def CDTrainer(config):

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
    optimizer_G = optim.AdamW(net_G.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    exp_lr_scheduler_G = get_scheduler(optimizer_G,500) ###200
    #exp_lr_scheduler_G = get_scheduler(optimizer_G,100)

    running_metric1 = ConfuseMatrixMeter(n_class=4)     ###### 4 classes
    running_metric2 = ConfuseMatrixMeter(n_class=4)    


    #checkpoint = session.get_checkpoint()             #### checkpoint for ray_tune
    #timer = Timer()

    #### batch_size
    train_dataset,val_dataset,test_dataset = load_dataset()
    
    #trainloader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8)
    trainloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8,worker_init_fn=seed_worker,generator=gen)
    valloader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8)

    dataloaders = {'train':trainloader, 'val':valloader}
    print(len(dataloaders))


    #  training log
    epoch_acc = 0
    epoch_loss = 0     #### loss

    best_val_loss = 0  ####

    best_epoch_id = 0
    epoch_to_start = 0
    max_num_epochs = 500
    #max_num_epochs = 4

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
            if epoch_id>=0: #0(reg)  #1000(no reg)   ##### Epoch to apply regularization
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
               lam =config['lam']
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
        print(train_reg / train_steps)
        #print('\n')

        ################## Eval ##################
        ##########################################
        running_metric1.clear()
        running_metric2.clear()
        net_G.eval()

        val_loss=0
        val_loss1=0
        val_loss2=0
        val_steps=0

        # Iterate over data.
        for batch_id, batch in enumerate(dataloaders['val'], 0):
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
            current_score1 = running_metric1.update_cm(pr=G_pred1.cpu().numpy(), gt=target1.cpu().numpy())

            target2 = batch['L2'].to(device).detach()
            G_pred2 = G_pred2.detach()
            G_pred2 = torch.argmax(G_pred2, dim=1)
            current_score2 = running_metric2.update_cm(pr=G_pred2.cpu().numpy(), gt=target2.cpu().numpy())


        scores1 = running_metric1.get_scores()
        val_acc1 = scores1['mf1']

        scores2 = running_metric2.get_scores()
        val_acc2 = scores2['mf1']

        mean_acc = (val_acc1+val_acc2)/2

        print('validation:')
        print('epoch',epoch_id)
        print(val_loss / val_steps)
        print(val_acc1)
        print(val_acc2)
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

            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                {   'model_G_state_dict': net_G.state_dict(),
                    'optimizer_G_state_dict':optimizer_G.state_dict(),
                    'exp_lr_scheduler_G_state_dict': exp_lr_scheduler_G.state_dict()}, path
            )

            # Send the current training result back to Tune
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report({"loss": val_loss / val_steps, "f1_1": val_acc1, "f1_2": val_acc2, "f1":mean_acc}, checkpoint=checkpoint)


