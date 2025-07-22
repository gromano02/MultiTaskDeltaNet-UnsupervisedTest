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
from models.load_unet_s import load_dataset_s

from utils import de_norm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import utils


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

    running_metric1 = ConfuseMatrixMeter(n_class=4)    #### pair metric
    running_metric2 = ConfuseMatrixMeter(n_class=2)    #### label metric

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


    ##### Unet prediction
    ### Validation:
    unet_pred=[]
    for batch_id, batch in enumerate(dataloaders['val'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             unet_pred.append(pred)

    print('unet_pred val:',len(unet_pred))


    ##### make image pair based on differrent region
    test_label=[]

    ##### 103:46, 301:18
    ##### 201:27, 203:14
    for m in range(46):
            numb = 46

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

            numb = 46+18

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
    #### change detection: 
    for batch_id, batch in enumerate(val_pair_load, 0):

        target1 = batch['L1'].to(device).detach()
        target1 = np.squeeze(target1)
        G_pred1 = test_label[batch_id]
        current_score1 = running_metric1.undate_score(pr=G_pred1, gt=target1.cpu().numpy())

        ##### individual image f1 score: 
        ##message = '%s _A1: ,' % (batch_id)
        ##for k, v in current_score1.items():
        ##    message += '%s: %.5f ' % (k, v)
        ##print(message)

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    print('validation results:')
    print('Overall change detection:')
    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ##### Unet prediction:
    #### label prediction:
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

             target=img_gt.detach()
             target = np.squeeze(target)

             current_score2 = running_metric2.undate_score(pr=pred, gt=target.cpu().numpy())

             ##message = '%s _A1: ,' % (batch_id)
             ##for k, v in current_score2.items():
             ##    message += '%s: %.5f ' % (k, v)
             ##print(message) 

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']
    print('Overall lable (segmentation) prediction:')
    #print(val_acc2)

    #print('model save at:',epoch_save)
    message = 'A1_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)
    print('\n')



    running_metric1.clear()
    running_metric2.clear()
    ##### Unet prediction
    ###  Test: 
    unet_pred=[]
    for batch_id, batch in enumerate(dataloaders['test'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             unet_pred.append(pred)

    print('unet_pred test:',len(unet_pred))


    ##### make image pair based on differrent region
    test_label=[]

    ##### 103:46, 301:18
    ##### 201:27, 203:14
    for m in range(27):
            numb = 27

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


    for m in range(27,27+14):

            numb = 27+14 

            for n in range(27,numb):

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
    #### change detection: 
    for batch_id, batch in enumerate(test_pair_load, 0):

        target1 = batch['L1'].to(device).detach()
        target1 = np.squeeze(target1)
        G_pred1 = test_label[batch_id]
        current_score1 = running_metric1.undate_score(pr=G_pred1, gt=target1.cpu().numpy())

        ##message = '%s _A1: ,' % (batch_id)
        ##for k, v in current_score1.items():
        ##    message += '%s: %.5f ' % (k, v)
        ##print(message)

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    print('test results:')
    print('Overall change detection:')
    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ##### Unet prediction:
    #### label prediction:
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

             target=img_gt.detach()
             target = np.squeeze(target)

             current_score2 = running_metric2.undate_score(pr=pred, gt=target.cpu().numpy())

             ##message = '%s _A1: ,' % (batch_id)
             ##for k, v in current_score2.items():
             ##    message += '%s: %.5f ' % (k, v)
             ##print(message) 

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']
    print('Overall lable (segmentation) prediction:')
    #print(val_acc2)

    #print('model save at:',epoch_save)

    message = 'A1_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)
    print('\n')    



    ###### seperate results: 
    val_dataset1,val_dataset2,test_dataset1,test_dataset2 = load_dataset_s()

    valloader1 = DataLoader(val_dataset1, batch_size=1, shuffle=False, num_workers=8)
    testloader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=8)
    valloader2= DataLoader(val_dataset2, batch_size=1, shuffle=False, num_workers=8)
    testloader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=8)

    dataloaders = {'val1':valloader1, 'val2':valloader2, 'test1':testloader1, 'test2':testloader2}


    running_metric1.clear()
    running_metric2.clear()

    ##### Unet prediction
    ####  103:
    unet_pred=[]
    for batch_id, batch in enumerate(dataloaders['val1'], 0):
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
    ##### 201:27, 203:14
    for m in range(46):
            numb = 46

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

    print('pred:',len(test_label))


    ##### Same order as the pair branch model:
    #### change detection: 
    for batch_id, batch in enumerate(val_pair_load, 0):
        
        if batch_id==2116:
            break

        target1 = batch['L1'].to(device).detach()
        target1 = np.squeeze(target1)
        G_pred1 = test_label[batch_id]
        current_score1 = running_metric1.undate_score(pr=G_pred1, gt=target1.cpu().numpy())

        ##message = '%s _A1: ,' % (batch_id)
        ##for k, v in current_score1.items():
        ##    message += '%s: %.5f ' % (k, v)
        ##print(message)

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    print('103 results:')
    print('Overall change detection:')
    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ##### Unet prediction:
    #### label prediction:
    for batch_id, batch in enumerate(dataloaders['val1'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)
             img_gt = batch['L1'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             pred = pred.astype(np.int64)

             target=img_gt.detach()
             target = np.squeeze(target)

             current_score2 = running_metric2.undate_score(pr=pred, gt=target.cpu().numpy())

             ##message = '%s _A1: ,' % (batch_id)
             ##for k, v in current_score2.items():
             ##    message += '%s: %.5f ' % (k, v)
             ##print(message) 

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']
    print('Overall lable (segmentation) prediction:')
    #print(val_acc2)

    print('model save at:',epoch_save)

    message = 'A1_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)
    print('\n')



    running_metric1.clear()
    running_metric2.clear()
    ##### Unet prediction
    ####  301:
    unet_pred=[]
    for batch_id, batch in enumerate(dataloaders['val2'], 0):
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
    ##### 201:27, 203:14
    for m in range(18):

            numb = 18

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

    print('pred:',len(test_label))


    ##### Same order as the pair branch model:
    #### change detection: 
    for batch_id, batch in enumerate(val_pair_load):

        if batch_id>=2116:
           #print('batch_id',batch_id) 
           target1 = batch['L1'].to(device).detach()
           target1 = np.squeeze(target1)
           G_pred1 = test_label[batch_id-2116]
           current_score1 = running_metric1.undate_score(pr=G_pred1, gt=target1.cpu().numpy())

           ##message = '%s _A1: ,' % (batch_id)
           ##for k, v in current_score1.items():
           ##    message += '%s: %.5f ' % (k, v)
           ##print(message)

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    print('301 results:')
    print('Overall change detection:')
    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ##### Unet prediction:
    #### label prediction:
    for batch_id, batch in enumerate(dataloaders['val2'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)
             img_gt = batch['L1'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             pred = pred.astype(np.int64)

             target=img_gt.detach()
             target = np.squeeze(target)

             current_score2 = running_metric2.undate_score(pr=pred, gt=target.cpu().numpy())

             ##message = '%s _A1: ,' % (batch_id)
             ##for k, v in current_score2.items():
             ##    message += '%s: %.5f ' % (k, v)
             ##print(message) 

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']
    print('Overall lable (segmentation) prediction:')
    #print(val_acc2)

    #print('model save at:',epoch_save)

    message = 'A1_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)
    print('\n') 



    running_metric1.clear()
    running_metric2.clear()

    ##### Unet prediction
    ####  201:
    unet_pred=[]
    for batch_id, batch in enumerate(dataloaders['test1'], 0):
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
    ##### 201:27, 203:14
    for m in range(27):

            numb = 27

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

    print('pred:',len(test_label))


    ##### Same order as the pair branch model:
    #### change detection: 
    for batch_id, batch in enumerate(test_pair_load, 0):

        if batch_id==729:
            break

        target1 = batch['L1'].to(device).detach()
        target1 = np.squeeze(target1)
        G_pred1 = test_label[batch_id]
        current_score1 = running_metric1.undate_score(pr=G_pred1, gt=target1.cpu().numpy())

        ##message = '%s _A1: ,' % (batch_id)
        ##for k, v in current_score1.items():
        ##    message += '%s: %.5f ' % (k, v)
        ##print(message)

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    print('201 results:')
    print('Overall change detection:')
    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ##### Unet prediction:
    #### label prediction:
    for batch_id, batch in enumerate(dataloaders['test1'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)
             img_gt = batch['L1'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             pred = pred.astype(np.int64)

             target=img_gt.detach()
             target = np.squeeze(target)

             current_score2 = running_metric2.undate_score(pr=pred, gt=target.cpu().numpy())

             ##message = '%s _A1: ,' % (batch_id)
             ##for k, v in current_score2.items():
             ##    message += '%s: %.5f ' % (k, v)
             ##print(message) 

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']
    print('Overall lable (segmentation) prediction:')
    #print(val_acc2)

    #print('model save at:',epoch_save)

    message = 'A1_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)
    print('\n')


    running_metric1.clear()
    running_metric2.clear()
    ##### Unet prediction
    ####  203:
    unet_pred=[]
    for batch_id, batch in enumerate(dataloaders['test2'], 0):
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
    ##### 201:27, 203:14
    for m in range(14):

            numb = 14

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

    print('pred:',len(test_label))


    ##### Same order as the pair branch model:
    #### change detection: 
    for batch_id, batch in enumerate(test_pair_load):

        if batch_id>=729:
           #print('batch_id',batch_id) 
           target1 = batch['L1'].to(device).detach()
           target1 = np.squeeze(target1)
           G_pred1 = test_label[batch_id-729]
           current_score1 = running_metric1.undate_score(pr=G_pred1, gt=target1.cpu().numpy())

           ##message = '%s _A1: ,' % (batch_id)
           ##for k, v in current_score1.items():
           ##    message += '%s: %.5f ' % (k, v)
           ##print(message)

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    print('203 results:')
    print('Overall change detection:')
    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ##### Unet prediction:
    #### label prediction:
    for batch_id, batch in enumerate(dataloaders['test2'], 0):
        with torch.no_grad():
             img_in = batch['A'].to(device)
             img_gt = batch['L1'].to(device)

             G_pred = net_G(img_in)

             pred = torch.argmax(G_pred, dim=1, keepdim=True)
             pred = utils.make_numpy_grid_lb(pred)

             pred = np.clip(pred, a_min=0.0, a_max=1.0)
             pred = pred.astype(np.int64)

             target=img_gt.detach()
             target = np.squeeze(target)

             current_score2 = running_metric2.undate_score(pr=pred, gt=target.cpu().numpy())

             ##message = '%s _A1: ,' % (batch_id)
             ##for k, v in current_score2.items():
             ##    message += '%s: %.5f ' % (k, v)
             ##print(message) 

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']
    print('Overall lable (segmentation) prediction:')
    #print(val_acc2)

    #print('model save at:',epoch_save)

    message = 'A1_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


