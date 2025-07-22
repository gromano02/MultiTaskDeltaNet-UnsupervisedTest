import numpy as np # linear algebra
import os

from models.trainer_unet import*

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CDTrainer()
