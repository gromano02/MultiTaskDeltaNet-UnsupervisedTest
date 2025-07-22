import numpy as np # linear algebra
import os

#from models.test_unet_cd_a1 import *
#from models.test_unet_cd_a2 import *
#from models.test_unet_cd import *

#from models.test_unet_hd_a1 import *
from models.test_unet_hd_a2 import *

#path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/save_model/unet/unet_testa1_results/'
#path = '/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/save_model/unet/unet_testa2_results/'
#path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/unet_same_test/a111_results_all_aug/'
#path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/unet_same_test/a222_results_all_aug/'
#path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/unet_same_test/a1_results/'
path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/cv_github/checkpoints/unet_same_test/a2_results/'

CDTest(path)
