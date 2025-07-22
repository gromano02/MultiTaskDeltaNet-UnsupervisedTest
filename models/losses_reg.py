import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def reg_phy(gt1,gt2,pred1,pred2,y1,y2,max_,min_,Len,Pi):

    m = nn.Softmax(dim=1)
    pred1 = m(pred1)
    pred2 = m(pred2)

    pred1 = pred1.permute(1,0,2,3)
    pred2 = pred2.permute(1,0,2,3)

    pred1 = pred1.reshape(4,-1)
    pred2 = pred2.reshape(4,-1)

    color1 = torch.sum(pred1, dim=1)
    color2 = torch.sum(pred2, dim=1)  

    red = color1[1]
    green = color1[2]
    grey = color1[3]

    a1_1 = grey+green ### pred_ref a1
    a2_1 = grey+red   ### pred_res a1


    red = color2[1]
    green = color2[2]
    grey = color2[3]

    a1_2 = grey+green ### pred_ref a2
    a2_2 = grey+red   ### pred_res a2

    #### V_pred:
    pi = Pi[0]
    L = Len[0]
    V1=pi*(a1_1**2-a1_2**2)/(4*L)
    V2=pi*(a2_1**2-a2_2**2)/(4*L)

    V1_norm = (V1-min_)/(max_-min_)
    V2_norm = (V2-min_)/(max_-min_)

    #pred = V1-V2
    pred = V1_norm-V2_norm    

    #### Avrami value:
    gt = torch.sum(y1-y2)

    dif = gt-pred
    #return torch.square(dif)/gt_size
    return torch.mean(torch.square(dif))    
