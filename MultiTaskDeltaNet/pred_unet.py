import numpy as np # linear algebra
import os

##### When you need to run the prediction for a1 or a2, 
##### change the corresponding line for a1 and a2. 

from models.test_unet_cd_a1 import *       #### for a1 unet prediction (model performance) 
#from models.test_unet_cd_a2 import *      #### for a2 unet prediction

#### for a1 unet prediction, location of saved a1 unet model
path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/unet/a1_results/'   

#### for a2 unet prediction, loation of saved a2 unet model  
#path ='/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/unet/a2_results/'

CDTest(path)


#### 
#from models.evaluator_unet import *      ##### model performance and prediction plots 
#CDEva()
