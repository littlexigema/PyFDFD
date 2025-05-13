import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from use_senior_code import *
from loss_fn import L_TV, L_TV_L1, L_TV_frac, L_TV_reg,EdgeEnhanceSmoothLoss
import matplotlib.pyplot as plt

from config_fresnel import *
from Forward import Forward_model
classes = "mnist"#'AU'#'multi_shapes'
name_ = '3/'#'1/'#'ground_truth.npy'#'1/gt.npy'
FWD = Forward_model(inverse=True)
FWD.get_system_matrix(fre)#主要是计算A_for
FWD.field.set_chi(load_from_gt=True,file = f'./Data/{classes}/{name_}gt.npy')
FWD.field.set_scatter_E_Born_Appro(FWD.A)

import pandas as pd
pwd = os.getcwd()
path = os.path.join(pwd,'Data/fresnel-1')
data = pd.read_csv(os.path.join(path,'twodielTM_4f.txt'))
