import torch
import torch.nn as nn
from config import C_0,C_norm,xs,ys,m_unit,R_receiver,R_transmitter,n_T,n_R

class CSI_model(nn.Module):
    def __init__(self, fre):
        super().__init__()
        self.lambda_ = C_0/(fre*1e9)#量纲连续
        self.wvlen = self.lambda_/m_unit#量纲离散
        self.omega_continuous =  2*torch.pi*fre
        self.omega_discrete = 2*torch.pi*C_norm/self.wvlen
        self.contrast = torch.empty((xs,ys),dtype = torch.complex64)#对比度函数
        self.J = torch.empty((xs,ys),dtype = torch.complex64)#对比源J = \chi .* E
        theta_T = [num*(360/n_T)for num in range(n_T)]
        theta_R = [num*(360/n_R)for num in range(n_R)]
        
        self.theta_T = torch.pi/180*torch.tensor(theta_T).reshape(-1,1)
        self.theta_R = torch.pi/180*torch.tensor(theta_R).reshape(-1,1)
        self.pos_T = torch.concat([torch.cos(theta_T),torch.sin(theta_T)],dim=1)*R_transmitter
        self.pos_R = torch.concat([torch.cos(theta_R),torch.sin(theta_R)],dim=1)*R_receiver
        self.E_R = torch.empty((n_R,n_T),dtype = torch.complex64) 
        self.E_inc = torch.empty((xs,ys),dtype = torch.complex64)
        self.E_scat = torch.empty((xs,ys),dtype = torch.complex64)
        self.E_tot = torch.empty((xs,ys),dtype = torch.complex64)

        self.contrast = torch.empty((xs,ys),dtype = torch.complex64)

        self.phi = torch.empty((n_R,xs*ys),dtype = torch.complex64)#M_d A^-1,yee网格离散化(maybe)
        self.A = None#稀疏矩阵占用大量空间，使用前不初始化

    def set_contrast():
        pass
    

        