import numpy as np # linear algebra
import os

from models.trainer_unet import*

##### When you need to run the prediction for a1 or a2, 
##### change thecorresponding line for a1 and a2 at trainer_unet.py 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CDTrainer()
