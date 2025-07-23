import numpy as np
import matplotlib.pyplot as plt
import os
import re
from PIL import Image

from models.nets import *
from models.test_siam_ablation import *
from misc.metric_tool import ConfuseMatrixMeter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils import de_norm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import utils

##### When you need to run the prediction for a1 or a2,
##### change the corresponding line for a1 and a2.

def normalize(input_data):
    return (input_data.astype(np.float32))/255.0

def denormalize(input_data):
    input_data = (input_data) * 255
    return input_data.astype(np.uint8)

palette = np.array([[1, 1, 1],  # "no change"
                    [0.5, 0, 0],  # "appearing(1)"
                    [0, 0.5, 0],  # "disappearing(-1)"
                    [0.4,0.4,0.5]], dtype='float32')  # "overlap(2)"

def CDEva(model_path,res_path,names):  ### name:val or test

    path='/home/yun13001/dataset/Carbon/tianyu_new_data/New_distribution/'
    dirnames = os.listdir(path)
    dirnames.remove('kinetic_curves.xlsx')
    dirnames=sorted(dirnames, key=lambda s: float(re.findall(r'\d+', s)[0]))
    print(dirnames)

    file_names={}
    for dirname in dirnames:
        for root, dirnames, filenames_x in os.walk(path+'/'+dirname+'/results/img'):
            break
        filenames_x = sorted(filenames_x, key=lambda s: float(re.findall(r'\d+', s)[2]))
        if dirname == '201' or dirname == '203':
            filenames_x.pop(0)
        else:
            filenames_x.pop(0)
            filenames_x.pop(0)

        file_names[dirname]=filenames_x

    train_name=['102_R1','102_R2', '302']
    val_name = ['103','301']
    test_name= ['201','203']
    all_name=['103','301','201','203']

    if names == 'val':
        names = val_name
    if names == 'test':
        names = test_name

    net_G = Siamese(in_channels=3, out_channels=4, init_features=32)

    running_metric1 = ConfuseMatrixMeter(n_class=2)     ###### each dataset

    running_metric_all1 = ConfuseMatrixMeter(n_class=2)     ###### all dataset

    PATH = model_path+'best_a1.pt'
    #PATH = model_path+'best_a2.pt'
    checkpoint = torch.load(PATH)
    net_G.load_state_dict(checkpoint['model_G_state_dict'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G = nn.DataParallel(net_G)
    net_G.to(device)
    net_G.eval()

    running_metric1.clear()
    running_metric_all1.clear()


    for name in names:

        running_metric1.clear()

        ref=[]
        label1=[]
        label2=[]

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            x_ref = np.array(x_ref)
            x_label1 = np.array(x_label1)
            x_label2 = np.array(x_label2)

            ref.append(x_ref)
            label1.append(x_label1)
            label2.append(x_label2)


        ref = np.array(ref)
        label1 = np.array(label1)
        label2 = np.array(label2)

        label1 = denormalize(label1)
        label2 = denormalize(label2)

        a1_f1=[]
        a2_f1=[]
        name_save=[]

        for m in range(len(ref)-1,-1,-1):

            pred_img1=[]
            pred_img2=[]

            numb=m

            name_ = imgs[m]
            print(name_)

            name_save.append(name_)


            n=m
            name__ = imgs[n]

            img1 = ref[n]
            img2 = ref[len(ref)-1]

            img1 = TF.to_tensor(img1).to(device)
            img2 = TF.to_tensor(img2).to(device)
            img1 = TF.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img2 = TF.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

            G_pred1 = net_G(img1, img2)

            trans = nn.Softmax(dim=1)
            G_pred1 = trans(G_pred1)

            pred1 = utils.make_numpy_grid(G_pred1)

            pred1 = np.argmax(pred1, axis=2)

            ##### a1
            for l in range(pred1.shape[0]):
                for k in range(pred1.shape[1]):

                    if pred1[l,k]==0:
                        pred1[l,k]=1

                    if pred1[l,k]==2:
                        pred1[l,k]=0
                    if pred1[l,k]==3:
                        pred1[l,k]=0

            pred1 = np.stack([pred1, pred1, pred1], axis=-1)
            pred_img1.append(pred1)
            pred1 = pred1.astype(np.float64)

            try:
                os.makedirs(res_path+'a1_b_order')
            except OSError:
                pass

            try:
                os.makedirs(res_path+'a1_b_order/'+name)
            except OSError:
                pass

            vis_dir1 = res_path + 'a1_b_order/'+name

            file_name1 = os.path.join(vis_dir1, name_)
            plt.imsave(file_name1, pred1)

            target1 = label1[m]      ##### label1
            #target1 = label2[m]        ##### label2
            pred1 = pred1.astype(np.int64)
            current_score1 = running_metric1.undate_score(pr=pred1, gt=target1)
            current_score_all1 = running_metric_all1.undate_score(pr=pred1, gt=target1)

            message = 'a1_'+'_'+ name_+', '
            #message = 'a2_'+'_'+ name_+', '
            for k, v in current_score1.items():
                message += '%s: %.5f ' % (k, v)
            print(message)
            a1_value=current_score1['mf1']
            a1_f1.append(a1_value)


        ######### Each dataset performance:
        print(name+'_performance:')
        scores1 = running_metric1.get_scores()


        message = 'A1_'+name
        #message = 'A2_'+name
        for k, v in scores1.items():
            message += '%s: %.5f ' % (k, v)
        print(message)


    ##### Overall performance:
    print('Overall Performance:')
    scores1 = running_metric_all1.get_scores()

    message = 'A1_'
    #message = 'A2_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


