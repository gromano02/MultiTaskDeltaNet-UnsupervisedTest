import numpy as np
import matplotlib.pyplot as plt
import os

from models.nets import *
from models.sia_two_out_four_class import Siamese        ##### for no_init
#from models.siamese_with_weights import Siamese         ##### for init1 or init2, change the corresponding line for init1 or init2 siamese_with_weights.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import get_alpha, get_alpha1, get_alpha2, softmax_helper, FocalLoss, mIoULoss, mmIoULoss, reg_term
from models.load_cv import load_dataset
from models.load_cv_s import load_dataset_s

from utils import de_norm
from sklearn.metrics import confusion_matrix
import utils

def CDTest(path):

    #### batch_size
    train_dataset,val_dataset,test_dataset = load_dataset()

    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    dataloaders = {'train':trainloader, 'val':valloader, 'test':testloader}

    net_G1 = Siamese(in_channels=3, out_channels=4, init_features=32)
    net_G2 = Siamese(in_channels=3, out_channels=4, init_features=32)
    net_G3 = Siamese(in_channels=3, out_channels=4, init_features=32)

    running_metric1 = ConfuseMatrixMeter(n_class=4)     ###### 4 classes
    running_metric2 = ConfuseMatrixMeter(n_class=4)

    PATH1 = path+'best_a1.pt'
    PATH2 = path+'best_a2.pt'
    PATH3 = path+'best_m.pt'

    #### single gpu load model
    ### model_a1
    checkpoint = torch.load(PATH1,weights_only=False)
    net_G1.load_state_dict(checkpoint['model_G_state_dict'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G1 = torch.nn.DataParallel(net_G1)
    net_G1.to(device)
    net_G1.eval()

    ### model_a2
    checkpoint = torch.load(PATH2,weights_only=False)
    net_G2.load_state_dict(checkpoint['model_G_state_dict'])
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G2 = torch.nn.DataParallel(net_G2)
    net_G2.to(device)
    net_G2.eval()

    ### model_m
    checkpoint = torch.load(PATH3,weights_only=False)
    net_G3.load_state_dict(checkpoint['model_G_state_dict'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G3 = torch.nn.DataParallel(net_G3)
    net_G3.to(device)
    net_G3.eval()



    ###### Validation
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['val'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G1(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()


            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('Validation:')
    print('model save a1:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['val'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G2(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()


            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('Validation:')
    print('model save a2:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['val'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G3(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()


            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('Validation:')
    print('model save mean(a1,a2):')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)




    ###### Testing

    ### a1 model
    running_metric1.clear()
    running_metric2.clear()

    # Iterate over data.
    for batch_id, batch in enumerate(dataloaders['test'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G1(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()
           
            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())
             

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']
    
    print('Testing:')
    print('model save a1:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ### a2 model 
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['test'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G2(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('model save a2:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ### mean model 
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['test'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G3(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('model save mean(a1,a2):')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ###### seperate results: 
    val_dataset1,val_dataset2,test_dataset1,test_dataset2 = load_dataset_s()

    valloader1 = DataLoader(val_dataset1, batch_size=1, shuffle=False, num_workers=8)
    testloader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=8)
    valloader2= DataLoader(val_dataset2, batch_size=1, shuffle=False, num_workers=8)
    testloader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, num_workers=8)

    dataloaders = {'val1':valloader1, 'val2':valloader2, 'test1':testloader1, 'test2':testloader2}


    #### 103:
    ### model a1
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['val1'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G1(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('val 103:')
    print('model save a1:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ### model a2
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['val1'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G2(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('val 103:')
    print('model save a1:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ### model mean
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['val1'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G3(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('val 103:')
    print('model save mean(a1,a2):')

    message = 'A1_'      
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():   
        message += '%s: %.5f ' % (k, v)
    print(message)


    #### 301:
    ### model a1
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['val2'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G1(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('val 301:')
    print('model save a1:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ### model a2
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['val2'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G2(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('val 301:')
    print('model save a2:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ### model mean
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['val2'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G3(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('val 301:')
    print('model save mean(a1,a2)::')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    #### test1 (201):
    ### model a1
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['test1'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G1(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('test 201:')
    print('model save a1:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ### model a2
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['test1'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G2(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('test 201:')
    print('model save a2:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ### model mean
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['test1'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G3(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('test 201:')
    print('model save mean(a1,a2):')

    message = 'A1_'      
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():   
        message += '%s: %.5f ' % (k, v)
    print(message)


    #### test2(203):
    ### model a1
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['test2'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G1(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('test 203:')
    print('model save a1:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


    ### model a2
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['test2'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G2(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('test 203:')
    print('model save a2:')

    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)



    ### model mean
    # Iterate over data.
    running_metric1.clear()
    running_metric2.clear()
    for batch_id, batch in enumerate(dataloaders['test2'], 0):

        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred1, G_pred2 = net_G3(img_in1, img_in2)

            #####Backward
            gt1 = batch['L1'].to(device).detach()
            gt2 = batch['L2'].to(device).detach()

            G_pred1 = torch.argmax(G_pred1, dim=1, keepdim=True)
            G_pred2 = torch.argmax(G_pred2, dim=1, keepdim=True)

            G_pred1 = G_pred1.detach()
            G_pred2 = G_pred2.detach()      

            current_score1 = running_metric1.undate_score(pr=G_pred1.cpu().numpy(), gt=gt1.cpu().numpy())
            current_score2 = running_metric2.undate_score(pr=G_pred2.cpu().numpy(), gt=gt2.cpu().numpy())

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']

    scores2 = running_metric2.get_scores()
    val_acc2 = scores2['mf1']

    print('test 203:')
    print('model save mean(a1,a2):')

    message = 'A1_'      
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():  
        message += '%s: %.5f ' % (k, v)
    print(message)

