import pandas as pd
from config import *
import torch
from Field import Field

if __name__=="__main__":
    theta_T = torch.linspace(0,2*pi,n_T+1)[:-1]
    theta_R = torch.linspace(0,2*pi,n_R+1)[:-1]
    field = Field()
    field.set_incident_E(omega,theta_T,theta_R,R_transmitter,R_receiver)
    field.set_phi()
    field.set_chi(omega,name)
    field.set_scatter_E()
    E_R = field.E_R.T.reshape(-1)
    E_tot = field.E_tot.T.reshape(-1)
    E_R_real,E_R_imag = E_R.real,E_R.imag
    E_tot_real,E_tot_imag = E_tot.real,E_tot.imag
    """
        数据集前三行transmitter索引，receiver索引，fre GHZ
    """
    index_trans = torch.tensor([[i+1]*n_R for i in range(n_T)]).view(-1)
    index_rece  = torch.tensor(list(range(1,n_R+1))*n_T).view(-1)
    index_fre   = torch.zeros_like(index_trans)+fre
    df = pd.DataFrame({
        'index_trans'   : index_trans.numpy(),
        'index_rece'    : index_rece.numpy(),
        'fre'           : index_fre.numpy(),
        'E_tot_real'    : E_tot_real.numpy(),
        'E_tot_imag'    : E_tot_imag.numpy(),
        'E_R_real'      : E_R_real.numpy(),
        'E_R_imag'      : E_R_imag.numpy()
    })
    df.to_csv('{}.csv'.format(name),index = False)
    # print(field.E_R.shape)
    # print(field.E_scat.shape)