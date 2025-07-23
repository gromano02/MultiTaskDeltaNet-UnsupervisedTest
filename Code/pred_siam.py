import numpy as np # linear algebra
import os

#### segmentation value
from models.eva_back_all import*
#from models.eva_consec_all import*
#from models.eva_ensam_all import*
#from models.eva_for_all import*

##### no_init:
model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/no_init/'          #### model location
res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/no_init/back/'      #### prediction location 
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/no_init/consec/'
#res_path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/no_init/ensam/'
#res_path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/no_init/for/'

##### init2:
#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/init/init2/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/init2/back/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/init2/consec/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/init2/ensam/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/init2/for/'

##### init1:
#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/init/init1/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/init1/back/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/init1/consec/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/init1/ensam/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/results/init1/for/'

####ablation:
# model oneout
#from models.eva_back_all_ablation import*
#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/ablation/single_task_a1/'
#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/ablation/single_task_a2/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/ablation/single_task_a1/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/vis/ablation/single_task_a2/'



#names = 'val'
names = 'test'
CDEva(model_path,res_path,names)


#### Change detection value
#from models.test_siam_cd import *
#CDTest(model_path)




