import numpy as np
import matplotlib.pyplot as plt
import os
import re
from PIL import Image

from models.unet import UNet

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


from misc.metric_tool import ConfuseMatrixMeter
from models.load_unet_phy import load_dataset_phy
from models.load_unet import load_dataset

from utils import de_norm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import utils


##### When you need to run the prediction for a1 or a2, 
##### change thecorresponding line for a1 and a2. 


def normalize(input_data):
    return (input_data.astype(np.float32))/255.0

def denormalize(input_data):
    input_data = (input_data) * 255
    return input_data.astype(np.uint8)

palette = np.array([[1, 1, 1],  # "no change"
                    [0.5, 0, 0],  # "appearing(1)"
                    [0, 0.5, 0],  # "disappearing(-1)"
                    [0.4,0.4,0.5]], dtype='float32')  # "overlap(2)"  [0.4,0.4,0.5]

def CDEva():

    path='/home/yun13001/dataset/Carbon/tianyu_new_data/New_distribution/'    ### data location

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
    #test_name = ['201']
    #test_name = ['203']

    net_G = UNet(in_channels=3, out_channels=2, init_features=32)

    #PATH ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/unet/a1_results/best_a1.pt'
    PATH ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/Code/checkpoints/unet/a1_results/best_a1.pt'
    #PATH ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/unet/a2_results/best_a2.pt' 

    running_metric1 = ConfuseMatrixMeter(n_class=2)
    running_metric2 = ConfuseMatrixMeter(n_class=2)

  
    #### single gpu load model
    checkpoint = torch.load(PATH,weights_only=False)
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    epoch_save = checkpoint['epoch']

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G = torch.nn.DataParallel(net_G)
    net_G.to(device)
    net_G.eval()    

    #for name in val_name:    #### Loop all the folders
    for name in test_name:

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

        f1=[]
        name_save=[]

        for m in range(len(ref)):

            name = imgs[m]

            name_save.append(name)

            img1 = ref[m]
            img1 = TF.to_tensor(img1).to(device)
            img1 = TF.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img1 = img1.unsqueeze(0)

            G_pred = net_G(img1)

            pred = torch.argmax(G_pred, dim=1, keepdim=True)
            pred1 = utils.make_numpy_grid_lb(pred)

            pred1 = np.stack([pred1, pred1, pred1], axis=-1)
            pred1 = pred1.astype(np.float64)
            #vis_dir1 = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/unet/a1'    #### save the plots
            vis_dir1 = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/Code/vis/results/unet/a1'
            #vis_dir2 = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/unet/a2'
            file_name1 = os.path.join(vis_dir1, name)
            #file_name2 = os.path.join(vis_dir2, name)
            plt.imsave(file_name1, pred1)
            #plt.imsave(file_name2, pred1)

            target1 = label1[m]
            #target1 = label2[m]
            pred1 = pred1.astype(np.int64)
            current_score1 = running_metric1.undate_score(pr=pred1, gt=target1)
            index_name = name.split("_")[2]
            index = index_name.split('.')[0]
            message = 'a2_'+'_'+ name+', '
            for k, v in current_score1.items():
                message += '%s: %.5f ' % (k, v)
            print(message)

            f1_value=current_score1['mf1']
            f1.append(f1_value)
        
        f1=np.array(f1)
        name_save = np.array(name_save) 

        combine = np.array([[a,str(b)] for a, b in zip(name_save, f1)])

        name_index = name.split("_")[0]
        np.savetxt(vis_dir1+'/'+'a1_'+name_index+".txt", combine, fmt="%s", delimiter=",")
        #np.savetxt(vis_dir1+'/'+'a2_'+name_index+".txt", combine, fmt="%s", delimiter=",")

    scores1 = running_metric1.get_scores()
    val_acc1 = scores1['mf1']
    print('test:')
    print(val_acc1)

    message = 'A1_'
    #message = 'A2_'
    for k, v in scores1.items():
        message += '%s: %.5f ' % (k, v)
    print(message)
