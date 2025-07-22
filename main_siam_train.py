import numpy as np # linear algebra
import os

#from models.trainer_siam import*
#from models.trainer_siam2 import*
#from models.trainer_siam3 import*
#from models.trainer_siam_ablation import*
from models.trainer_siam_savemodel import*

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
CDTrainer()
