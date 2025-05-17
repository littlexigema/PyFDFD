"""
Author: Shaorui Guo
Time: 2025-05-09
Function: Example generate E-field and Green matrix using MOM
Attention: You should set configuration in config.py first
"""
from Forward import Forward_model
from config import * 
import os

pwd = os.getcwd()

FWD = Forward_model(inverse=True)
FWD.field.set_chi(load_from_gt=True,file = os.path.join(pwd,'generate_data_MOM','ms_9_gt.npy'))#automatically run set_incident_E_MOM()
FWD.field.set_scatter_E_MOM(omega=omega)
FWD.field.export_npy()
