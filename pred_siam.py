import numpy as np # linear algebra
import os

#from models.evaluator_siam import *

#from models.evaluator_back_order import*
#from models.evaluator_consecutive import*
#from models.evaluator_ensamble import*
#from models.evaluator_forward_order import*

#CDEva()


#### segmentation value
#from models.eva_back_all import*
#from models.eva_consec_all import*
#from models.eva_ensam_all import*
from models.eva_for_all import*

#from models.eva_back_hd import*
#from models.eva_consec_hd import*
#from models.eva_ensam_hd import*
#from models.eva_for_hd import*


#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/cv_all_pair_test/test1/'  #### cv_all_pair
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/cv_all_pair/back/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/cv_all_pair/consec/'
#res_path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/cv_all_pair/ensam/'
#res_path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/cv_all_pair/for/'

#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/siam_unet_test/siam_unet3_a1_10epoch_t2/'  ##### siam_unet3
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet3/back/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet3/consec/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet3/ensam/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet3/for/'

#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/siam_unet_test/model_save/siam_unet4/'     ######siam_unet4
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet4/back/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet4/consec/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet4/ensam/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet4/for/'

#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/siam_unet_test/model_save/siam_unet2_t1/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet2/back/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet2/consec/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet2/ensam/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet2/for/'


model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/siam_unet_test/siam_unet1/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet1/back/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet1/consec/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet1/ensam/'
res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/siam_unet1/for/'

####ablation
# model save
#model_path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/ablation/save_model2/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/ablation/save_model/'

# model oneout
#from models.eva_back_all_ablation import*
#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/ablation/out1/'
#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/ablation/out2/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/ablation/out1/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/ablation/out2/'



#names = 'val'
names = 'test'
#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/save_model/cv_test/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/test/'
CDEva(model_path,res_path,names)

#CDEva()

#### Change detection value
#from models.test_siam_cd3 import *
##path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/cv_all_pair_test/test1/'
##path='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/siam_unet_test/siam_unet3_a1_10epoch/'
#path='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/siam_unet_test/siam_unet1_t2/'
#path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/siam_unet_test/siam_unet1/'
#CDTest(path)




#from models.eva_back_cosnet import*
#from models.eva_cosnet import*
#names = 'val'
#names = 'test'
#model_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/save_model/cosnet/'
#res_path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/vis/results6/cosnet/back/'
#cosnetEva(model_path,res_path,names)
