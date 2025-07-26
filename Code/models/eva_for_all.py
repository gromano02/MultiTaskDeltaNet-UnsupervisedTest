import numpy as np
import matplotlib.pyplot as plt
import os
import re
from PIL import Image

from models.nets import *
from models.sia_two_out_four_class import Siamese          ##### for no_init
#from models.siamese_with_weights import Siamese             ##### for init1 or init2, change the corresponding line for init1 or init2 at siamese_with_weights.py
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


def normalize(input_data):
    return (input_data.astype(np.float32))/255.0

def denormalize(input_data):
    input_data = (input_data) * 255
    return input_data.astype(np.uint8)

palette = np.array([[1, 1, 1],  # "no change"
                    [0.5, 0, 0],  # "appearing(1)"
                    [0, 0.5, 0],  # "disappearing(-1)"
                    [0.4,0.4,0.5]], dtype='float32')  # "overlap(2)"


def CDEva(model_path,res_path,names):

    path='/home/yun13001/dataset/Carbon/tianyu_new_data/New_distribution/'

    dirnames = os.listdir(path)
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
    #val_name = ['103']
    #val_name = ['301']
    test_name= ['201','203']
    #test_name=['201']
    #test_name=['203']

    if names == 'val':
        names = val_name
    if names == 'test':
        names = test_name

    net_G = Siamese(in_channels=3, out_channels=4, init_features=32)

    running_metric1 = ConfuseMatrixMeter(n_class=2)     ###### each dataset
    running_metric2 = ConfuseMatrixMeter(n_class=2)

    running_metric_all1 = ConfuseMatrixMeter(n_class=2)     ###### all dataset
    running_metric_all2 = ConfuseMatrixMeter(n_class=2)


    PATH = model_path+'best_m.pt'
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
    running_metric2.clear()
    running_metric_all1.clear()
    running_metric_all2.clear()


    for name in names:
        running_metric1.clear()
        running_metric2.clear()

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

        ##### for forward: 
        for m in range(len(ref)):

            pred_img1=[]
            pred_img2=[]

            numb=m
            name_ = imgs[m]
            print(name_)
            name_save.append(name_)

            if numb!=0:

                n=m
                name__ = imgs[n]

                img1 = ref[n]
                img2 = ref[0]
                img1 = TF.to_tensor(img1).to(device)
                img2 = TF.to_tensor(img2).to(device)
                img1 = TF.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                img2 = TF.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)

                G_pred1, G_pred2 = net_G(img1, img2)

                trans = nn.Softmax(dim=1)
                G_pred1 = trans(G_pred1)
                G_pred2 = trans(G_pred2)

                pred1 = utils.make_numpy_grid(G_pred1)
                pred2 = utils.make_numpy_grid(G_pred2)

                pred1 = np.argmax(pred1, axis=2)
                pred2 = np.argmax(pred2, axis=2)

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
                   os.makedirs(res_path+'a1_f_order')
                except OSError:
                   pass

                try:
                   os.makedirs(res_path+'a1_f_order/'+name)
                except OSError:
                   pass

                vis_dir1 = res_path + 'a1_f_order/'+name

                file_name1 = os.path.join(vis_dir1, name_)
                plt.imsave(file_name1, pred1)


                target1 = label1[m]
                pred1 = pred1.astype(np.int64)
                current_score1 = running_metric1.undate_score(pr=pred1, gt=target1)
                current_score_all1 = running_metric_all1.undate_score(pr=pred1, gt=target1)

                message = 'a1_'+'_'+ name_+', '
                for k, v in current_score1.items():
                    message += '%s: %.5f ' % (k, v)
                print(message)
                a1_value=current_score1['mf1']
                a1_f1.append(a1_value)


                ##### a2
                for l in range(pred2.shape[0]):
                    for k in range(pred2.shape[1]):

                        if pred2[l,k]==0:
                            pred2[l,k]=1

                        if pred2[l,k]==2:
                            pred2[l,k]=0
                        if pred2[l,k]==3:
                            pred2[l,k]=0

                pred2 = np.stack([pred2, pred2, pred2], axis=-1)
                pred_img2.append(pred2)
                pred2 = pred2.astype(np.float64)

                try:
                   os.makedirs(res_path+'a2_f_order')
                except OSError:
                   pass

                try:
                   os.makedirs(res_path+'a2_f_order/'+name)
                except OSError:
                   pass

                vis_dir2 = res_path + 'a2_f_order/'+name

                file_name2 = os.path.join(vis_dir2, name_)
                plt.imsave(file_name2, pred2)

                target2 = label2[m]
                pred2 = pred2.astype(np.int64)
                current_score2 = running_metric2.undate_score(pr=pred2, gt=target2)
                current_score_all2 = running_metric_all2.undate_score(pr=pred2, gt=target2)

                message = 'a2_'+'_'+ name_+', '
                for k, v in current_score2.items():
                    message += '%s: %.5f ' % (k, v)
                print(message)
                a2_value=current_score2['mf1']
                a2_f1.append(a2_value)



            else:

                n=0
                name__ = imgs[n]

                img1 = ref[n]
                img2 = ref[m]
                img1 = TF.to_tensor(img1).to(device)
                img2 = TF.to_tensor(img2).to(device)
                img1 = TF.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                img2 = TF.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)

                G_pred1, G_pred2 = net_G(img1, img2)

                trans = nn.Softmax(dim=1)
                G_pred1 = trans(G_pred1)
                G_pred2 = trans(G_pred2)

                pred1 = utils.make_numpy_grid(G_pred1)
                pred2 = utils.make_numpy_grid(G_pred2)

                pred1 = np.argmax(pred1, axis=2)
                pred2 = np.argmax(pred2, axis=2)

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
                   os.makedirs(res_path+'a1_f_order')
                except OSError:
                   pass

                try:
                   os.makedirs(res_path+'a1_f_order/'+name)
                except OSError:
                   pass

                vis_dir1 = res_path + 'a1_f_order/'+name

                file_name1 = os.path.join(vis_dir1, name_)
                plt.imsave(file_name1, pred1)

                target1 = label1[m]
                pred1 = pred1.astype(np.int64)
                current_score1 = running_metric1.undate_score(pr=pred1, gt=target1)
                current_score_all1 = running_metric_all1.undate_score(pr=pred1, gt=target1)

                message = 'a1_'+'_'+ name_+', '
                for k, v in current_score1.items():
                    message += '%s: %.5f ' % (k, v)
                print(message)
                a1_value=current_score1['mf1']
                a1_f1.append(a1_value)


                ####### a2
                for l in range(pred2.shape[0]):
                    for k in range(pred2.shape[1]):

                        if pred2[l,k]==0:
                            pred2[l,k]=1

                        if pred2[l,k]==2:
                            pred2[l,k]=0
                        if pred2[l,k]==3:
                            pred2[l,k]=0

                pred2 = np.stack([pred2, pred2, pred2], axis=-1)
                pred_img2.append(pred2)
                pred2 = pred2.astype(np.float64)

                try:
                   os.makedirs(res_path+'a2_f_order')
                except OSError:
                   pass

                try:
                   os.makedirs(res_path+'a2_f_order/'+name)
                except OSError:
                   pass

                vis_dir2 = res_path + 'a2_f_order/'+name


                file_name2 = os.path.join(vis_dir2, name_)
                plt.imsave(file_name2, pred2)

                target2 = label2[m]
                pred2 = pred2.astype(np.int64)
                current_score2 = running_metric2.undate_score(pr=pred2, gt=target2)
                current_score_all2 = running_metric_all2.undate_score(pr=pred2, gt=target2)

                message = 'a2_'+'_'+ name_+', '
                for k, v in current_score2.items():
                    message += '%s: %.5f ' % (k, v)
                print(message)
                a2_value=current_score2['mf1']
                a2_f1.append(a2_value)


        a1_f1 = np.array(a1_f1)
        a2_f1 = np.array(a2_f1)
        name_save = np.array(name_save)

        combine1 = np.array([[a,str(b)] for a, b in zip(name_save, a1_f1)])
        combine2 = np.array([[a,str(b)] for a, b in zip(name_save, a2_f1)])

        name_index = name.split("_")[0]

        np.savetxt(vis_dir1+'/'+'a1_'+name+".txt", combine1, fmt="%s", delimiter=",")
        np.savetxt(vis_dir2+'/'+'a2_'+name+".txt", combine2, fmt="%s", delimiter=",")

        ######### Each dataset performance:
        print(name+'_performance:')
        scores1 = running_metric1.get_scores()
        scores2 = running_metric2.get_scores()

        val_acc1 = scores1['mf1']
        val_acc2 = scores2['mf1']

        print(val_acc1)
        print(val_acc2)

        message = 'A1_'+name
        for k, v in scores1.items():
            message += '%s: %.5f ' % (k, v)
        print(message)

        message = 'A2_'+name
        for k, v in scores2.items():
            message += '%s: %.5f ' % (k, v)
        print(message)


    ##### Overall performance:
    print('Overall Performance:')
    scores1 = running_metric_all1.get_scores()
    scores2 = running_metric_all2.get_scores()

    val_acc1 = scores1['mf1']
    val_acc2 = scores2['mf1']

    print('test:')
    print(val_acc1)
    print(val_acc2)


    message = 'A1_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)

    message = 'A2_'
    for k, v in scores2.items():
        message += '%s: %.5f ' % (k, v)
    print(message)


