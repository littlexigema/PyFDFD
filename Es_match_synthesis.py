"""FDFD计算散射场与实际散射场match"""

import torch
import numpy as np

img2mse = lambda x, y : torch.mean((x - y) ** 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
eta_0 = 120 * np.pi
c = 3e8
eps_0 = 8.85e-12
N_rec = 32#360#32#360  # Nb. of Receiver
N_inc = 16#8#16#8  # Nb. of Incidence

# i = torch.complex(0.0, 1)


MAX = 1
#MAX = 0.15/2
Mx = 64
step_size = 2 * MAX / (Mx - 1)
cell_area = step_size ** 2
from Forward import Forward_model
from config import *
FWD = Forward_model()
FWD.get_system_matrix(fre)
FWD.A = FWD.A.to(device)
FWD.field.R_mat = FWD.field.R_mat.to(device)
FWD.field.Phi_mat = FWD.field.Phi_mat.to(device)
FWD.field.E_inc = FWD.field.E_inc.to(device)
FWD.field.epsil = FWD.field.epsil.to(device)
FWD.field.Es = FWD.field.Es.to(device)
N_cell = 4096
epsilon = FWD.field.epsil
xi_all = -1j*omega * (epsilon - 1) * eps_0 * cell_area
xi_forward = torch.reshape(xi_all.t(), [-1, 1]).to(torch.complex64)
xi_forward_mat = torch.diag_embed(xi_forward.squeeze(-1))
# xi_E_inc = xi_forward_mat @ FWD.field.E_inc#xi_forward_mat = Diag(\epsilon)
J = xi_forward_mat@torch.linalg.inv(torch.eye(N_cell,device=device) - (FWD.field.Phi_mat @ xi_forward_mat))@FWD.field.E_inc
# J = \chi @ E_p = xi_forward_mat@ ...line 35
Es_pred = FWD.field.R_mat@torch.linalg.solve(FWD.A,FWD.osc.in_omega0()**2*J)
Es_true = FWD.field.Es

"""match scatter field"""
opposite = {1:17,2:19,3:21,4:23,5:25,6:27,7:29,8:31,9:1,10:3,11:5,12:7,13:9,14:11,15:13,16:15}
for col in range(N_inc):
    norm = Es_true[opposite[col+1]-1,col]/Es_pred[opposite[col+1]-1,col]
    print(norm)
    print('before:{}'.format(Es_pred[:,col]))
    Es_pred[:,col]*=norm
    print('after:{}'.format(Es_pred[:,col]))

# loss = (Es_pred-Es_true)/
img_loss_data = (img2mse(Es_pred.real, Es_true.real) + img2mse(Es_pred.imag, Es_true.imag))/torch.mean(Es_true.real **2 + Es_true **2)
print(img_loss_data)