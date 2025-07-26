import numpy as np # linear algebra
import os

from models.trainer_siam import*             ### for no_init, init1 or init2
#from models.trainer_siam_ablation import*   ### for Single-task

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CDTrainer()
