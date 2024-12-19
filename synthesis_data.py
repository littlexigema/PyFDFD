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
    field.set_chi(omega)
    field.set_scatter_E()
    E_R = field.E_R.T.reshape(-1)
    E_tot = field.E_tot.T.reshape(-1)
    E_R_real,E_R_imag = E_R.real,E_R.imag
    E_tot_real,E_tot_imag = E_tot.real,E_tot.imag
    print(field.E_R.shape)
    print(field.E_scat.shape)