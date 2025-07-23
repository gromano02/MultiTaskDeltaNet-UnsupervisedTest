import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
#def cross_entropy(input, target, weight=torch.tensor([0.01, 0.49, 0.50]), reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)



#Focal Loss
def get_alpha(supervised_loader):
    # get number of classes
    num_labels = 0
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        num_labels = max(max(list_unique),num_labels)
    num_classes = num_labels + 1
    # count class occurrences
    alpha = [0 for i in range(num_classes)]
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        l_unique_count = torch.stack([(label_batch.data==x_u).sum() for x_u in l_unique]) # tensor([65920, 36480])
        list_count = [count.item() for count in l_unique_count.flatten()]
        for index in list_unique:
            alpha[index] += list_count[list_unique.index(index)]
    return alpha


def get_alpha1(supervised_loader):
    # get number of classes
    num_labels = 0
    for batch in supervised_loader:
        label_batch = batch['L1']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        num_labels = max(max(list_unique),num_labels)
    num_classes = num_labels + 1
    # count class occurrences
    alpha = [0 for i in range(num_classes)]
    for batch in supervised_loader:
        label_batch = batch['L1']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        l_unique_count = torch.stack([(label_batch.data==x_u).sum() for x_u in l_unique]) # tensor([65920, 36480])
        list_count = [count.item() for count in l_unique_count.flatten()]
        for index in list_unique:
            alpha[index] += list_count[list_unique.index(index)]
    return alpha


def get_alpha2(supervised_loader):
    # get number of classes
    num_labels = 0
    for batch in supervised_loader:
        label_batch = batch['L2']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        num_labels = max(max(list_unique),num_labels)
    num_classes = num_labels + 1
    # count class occurrences
    alpha = [0 for i in range(num_classes)]
    for batch in supervised_loader:
        label_batch = batch['L2']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        l_unique_count = torch.stack([(label_batch.data==x_u).sum() for x_u in l_unique]) # tensor([65920, 36480])
        list_count = [count.item() for count in l_unique_count.flatten()]
        for index in list_unique:
            alpha[index] += list_count[list_unique.index(index)]
    return alpha


# for FocalLoss
def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=1, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
	
        alpha = self.alpha

        #print('len alpha:',len(alpha),'num_class:',num_class)

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
            alpha = 1/alpha # inverse of class frequency
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
	
        # to resolve error in idx in scatter_
        idx[idx==225]=0
        
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        #print('loss:',type(loss))    
        return loss


#miou loss
from torch.autograd import Variable
def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = torch.squeeze(tensor, dim=1).size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.type(torch.int64).view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.weights = Variable(weight)

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weights * inter) / (union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)

#Minimax iou
class mmIoULoss(nn.Module):
    def __init__(self, n_classes=2):
        super(mmIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        iou = inter/ (union + 1e-8)

        #minimum iou of two classes
        min_iou = torch.min(iou)

        #loss
        loss = -min_iou-torch.mean(iou)
        return loss


def reg_term(gt1,gt2,pred1,pred2):

    #print('gt shape:',gt1.shape)
    #print('pred shape:',pred1.shape)

    m = nn.Softmax(dim=1)
    pred1 = m(pred1)
    pred2 = m(pred2)
  
    pred1 = torch.argmax(pred1, dim=1)
    pred2 = torch.argmax(pred2, dim=1)

    pred1 = pred1.unsqueeze(1)
    pred2 = pred2.unsqueeze(1)

    gt1 =   gt1.permute(0,2,3,1)
    gt2 =   gt2.permute(0,2,3,1)
    pred1 = pred1.permute(0,2,3,1)
    pred2 = pred2.permute(0,2,3,1)

    #print('gt shape1:',gt1.shape)
    #print('pred shape1:',pred1.shape)

    (gt_unique1,gt_counts1) = torch.unique(gt1.reshape((-1,1)),dim=0,return_counts =True)
    (gt_unique2,gt_counts2) = torch.unique(gt2.reshape((-1,1)),dim=0,return_counts =True)

    (pred_unique1,pred_counts1) = torch.unique(pred1.reshape((-1,1)),dim=0,return_counts =True)
    (pred_unique2,pred_counts2) = torch.unique(pred2.reshape((-1,1)),dim=0,return_counts =True)    

    #### a1_gt
    white=0
    red = 0
    green = 0
    grey = 0
    color =[white,red,green,grey]
    for m in range(len(gt_unique1)):
        index = gt_unique1[m][0]
        #print(index)
        #print(counts[m])
        count = gt_counts1[m]
                
        color[index]= count 
        
    red = color[1]
    green = color[2]
    grey = color[3]
        
    #print(red,green,grey)
    
    a1_dif_gt = green-red
    a1_sum_gt = 2*grey+red+green

    #print(a_dif)
    #print(a_sum) 
    
    #### a2_gt
    white=0
    red = 0
    green = 0
    grey = 0
    color =[white,red,green,grey]
    for m in range(len(gt_unique2)):
        index = gt_unique2[m][0]
        count = gt_counts2[m]
        color[index]= count

    red = color[1]
    green = color[2]
    grey = color[3]

    a2_dif_gt = green-red
    a2_sum_gt = 2*grey+red+green
    
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

    a1_dif_pred = green-red
    a1_sum_pred = 2*grey+red+green    
    
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

    a2_dif_pred = green-red
    a2_sum_pred = 2*grey+red+green    

    #### (v1^2-v1'^2)-(v2^2-v2'^2)
    #### (a1^2-a1'^2) = (a1+a1')(a1-a1') 
    #### =(overlap+green+overlap+red)(overlap+green-overlap-red)
    #### =(2*overlap+red+green)(green-red)
    a1_gt = a1_sum_gt*a1_dif_gt
    a2_gt = a2_sum_gt*a2_dif_gt

    a1_pred = a1_sum_pred*a1_dif_pred
    a2_pred = a2_sum_pred*a2_dif_pred
    
    gt = a1_gt-a2_gt
    pred = a1_pred -a2_pred

    #print('reg:',type(pred))
    gt_size = gt1.size(dim=0)*gt1.size(dim=1)*gt1.size(dim=2)

    #print(gt_size)
    #return (gt-pred)
    #torch.square(gt-pred)
    print('dif:',gt-pred)
    #print('suqare dif:',torch.square(gt-pred))
    #print('abs dif:')

    #return torch.square(gt-pred)/gt_size
    return torch.abs(gt-pred)/gt_size 

"""
def reg_term_phy(gt1,gt2,pred1,pred2,Len,Pi):

    #print('gt shape:',gt1.shape)
    #print('pred shape:',pred1.shape)

    m = nn.Softmax(dim=1)
    pred1 = m(pred1)
    pred2 = m(pred2)

    pred1 = torch.argmax(pred1, dim=1)
    pred2 = torch.argmax(pred2, dim=1)

    pred1 = pred1.unsqueeze(1)
    pred2 = pred2.unsqueeze(1)

    gt1 =   gt1.permute(0,2,3,1)
    gt2 =   gt2.permute(0,2,3,1)
    pred1 = pred1.permute(0,2,3,1)
    pred2 = pred2.permute(0,2,3,1)

    #print('gt shape1:',gt1.shape)
    #print('pred shape1:',pred1.shape)

    (gt_unique1,gt_counts1) = torch.unique(gt1.reshape((-1,1)),dim=0,return_counts =True)
    (gt_unique2,gt_counts2) = torch.unique(gt2.reshape((-1,1)),dim=0,return_counts =True)

    (pred_unique1,pred_counts1) = torch.unique(pred1.reshape((-1,1)),dim=0,return_counts =True)
    (pred_unique2,pred_counts2) = torch.unique(pred2.reshape((-1,1)),dim=0,return_counts =True)

    #### a1_gt
    white=0
    red = 0
    green = 0
    grey = 0
    color =[white,red,green,grey]
    for m in range(len(gt_unique1)):
        index = gt_unique1[m][0]
        #print(index)
        #print(counts[m])
        count = gt_counts1[m]

        color[index]= count

    red = color[1]
    green = color[2]
    grey = color[3]

    #print(red,green,grey)

    a1_dif_gt = green-red
    a1_sum_gt = 2*grey+red+green

    #print(a_dif)
    #print(a_sum)


    #### a2_gt
    white=0
    red = 0
    green = 0
    grey = 0
    
    color =[white,red,green,grey]
    for m in range(len(gt_unique2)):
        index = gt_unique2[m][0]
        count = gt_counts2[m]
        color[index]= count

    red = color[1]
    green = color[2]
    grey = color[3]

    a2_dif_gt = green-red
    a2_sum_gt = 2*grey+red+green

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

    a1_dif_pred = green-red
    a1_sum_pred = 2*grey+red+green

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

    a2_dif_pred = green-red
    a2_sum_pred = 2*grey+red+green

    a1_gt = a1_sum_gt*a1_dif_gt
    a2_gt = a2_sum_gt*a2_dif_gt

    a1_pred = a1_sum_pred*a1_dif_pred
    a2_pred = a2_sum_pred*a2_dif_pred

    gt = a1_gt-a2_gt
    pred = a1_pred -a2_pred

    #print('reg:',type(pred))
    gt_size = gt1.size(dim=0)*gt1.size(dim=1)*gt1.size(dim=2)

    #print('dif:',gt-pred)

    pi =Pi[0]
    
    dif = gt-pred
    V = pi*dif/(4*Len[0])

    #return torch.square(gt-pred)/gt_size
    #return torch.abs(gt-pred)/gt_size
    return torch.abs(V)/gt_size
"""

def reg_term_phy(gt1,gt2,pred1,pred2,y1,y2,Len,Pi):

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
    
    pred = V1-V2
    #print('pred:',pred.shape)

    #### Avrami value:
    gt = torch.sum(y1-y2)
    #print('gt:',gt.shape)

    #### size (for average)
    gt_size = gt1.size(dim=0)*gt1.size(dim=1)*gt1.size(dim=2)

    dif = gt-pred     
    return torch.abs(dif)/gt_size 



