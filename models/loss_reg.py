import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def loss_term_phy(pred1,pred2,t1,t2,Len,Pi,max_,min_):

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
    #### y(t+delta_t)=t1 and y(t)=t2
    V1=pi*(torch.square(a1_1)-torch.square(a1_2))/(4*L) 
    V2=pi*(torch.square(a2_1)-torch.square(a2_2))/(4*L)

    pred = V1-V2
    
    #### y_dot 
    y_dot = pred/(t1-t2)
    #### Normalize y_dot, V1, V2
    y_dot = (y_dot-min_)/(max_-min_)
    y1_hat= (V1-min_)/(max_-min_)
    y2_hat= (V2-min_)/(max_-min_)

    #### y'
    n=1    ##### n is constant
    t_hat = (n/y_dot)*(y2_hat-1)*torch.log(y2_hat-1) 
    
    #### calculate the norm of batch, then mean of value
    dif = torch.mean(torch.square(t2-t_hat))

    return dif



def reg_phy(gt1,gt2,pred1,pred2,y1,y2,max_,min_,Len,Pi):

    m = nn.Softmax(dim=1)
    pred1 = m(pred1)
    pred2 = m(pred2)

    pred1 = torch.argmax(pred1, dim=1)
    pred2 = torch.argmax(pred2, dim=1)

    pred1 = pred1.unsqueeze(1)
    pred2 = pred2.unsqueeze(1)

    (pred_unique1,pred_counts1) = torch.unique(pred1.reshape((-1,1)),dim=0,return_counts =True)
    (pred_unique2,pred_counts2) = torch.unique(pred2.reshape((-1,1)),dim=0,return_counts =True)


    #### a1_pred
    white=0
    red = 0
    green = 0
    grey = 0
    color =[white,red,green,grey]
    for m in range(len(pred_unique1)):
        index = pred_unique1[m][0]
        count = pred_counts1[m]
        color[index]= count

    red = color[1]
    green = color[2]
    grey = color[3]

    #a1_dif_pred = green-red
    #a1_sum_pred = 2*grey+red+green
    a1_1 = grey+green ### pred_ref a1 
    a2_1 = grey+red   ### pred_res a1

    #### a2_pred
    white=0
    red = 0
    green = 0
    grey = 0
    color =[white,red,green,grey]
    for m in range(len(pred_unique2)):
        index = pred_unique2[m][0]
        count = pred_counts2[m]
        color[index]= count

    red = color[1]
    green = color[2]
    grey = color[3]

    #a2_dif_pred = green-red
    #a2_sum_pred = 2*grey+red+green
    a1_2 = grey+green ### pred_ref a2 
    a2_2 = grey+red   ### pred_res a2

    #### V_pred:
    #a1_pred = a1_sum_pred*a1_dif_pred
    #a2_pred = a2_sum_pred*a2_dif_pred
    #pred = a1_pred -a2_pred
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

    #### size (for average)
    #gt_size = gt1.size(dim=0)*gt1.size(dim=1)*gt1.size(dim=2)

    dif = gt-pred
    #return torch.square(gt-pred)/gt_size
    return torch.mean(torch.square(gt-pred))



