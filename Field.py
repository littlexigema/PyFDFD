from functions import hankel_0_1
from config import m_unit,R_transmitter,R_receiver,C_0,C_norm,pi,centre,xs,ys
from typing import Tuple,Union
from tqdm import trange
import torch
# import pandas as pd
# data = pd.read_csv('./data.csv',header=None).values
# data = torch.from_numpy(data)


class Field:
    m_unit = m_unit
    invdom = [centre[0]-xs,centre[0]+xs,centre[1]-ys,centre[1]+ys,centre[2],centre[2]]
    # xlchi  = round()
    nx     = int(xs*1e-3//m_unit)
    ny     = int(ys*1e-3//m_unit)
    
    def __init__(self) :
        """
        定义离散场类，存储多种离散场,
        receiver记录的E_mea,mesh_grid上的
        E_inc,E_scat,E_tot

        args:
            
        """
        
        pass
    
    def set_incident_E(self, omega, theta_T:torch.tensor,theta_R:torch.tensor,T_R:Union[torch.tensor,float],R_R:Union[torch.tensor,float] ) ->torch.Tensor:
        """
        对于给定的Transmitter的极坐标位置
        计算对于指定receiver位置和unit的入射场
        
        args:
            omega : 角频率,连续空间中的(与m的单位一致)
            pos_T (torch.Tensor): x,y位置(m)
            pos_R (torch.Tensor): each x,y位置(m)
        """

        assert isinstance(theta_T,torch.Tensor),TypeError("pos_T should be torch.Tensor!")
        assert isinstance(theta_R,torch.Tensor),TypeError("pos_R should be torch.Tensor!")
        if isinstance(T_R,Union[int,float]):
            T_R = torch.tensor(T_R,dtype=torch.float32)
        if isinstance(R_R,Union[int,float]):
            R_R = torch.tensor(R_R,dtype=torch.float32)
        # assert pos_T.shape[1]==2,RuntimeError("pos_T should have 2 columns!")
        # assert pos_R.shape[1]==2,RuntimeError("pos_R should have 2 columns!")
        """
            在参考CC-CSI代码发现,计算phi的时候,R的计算单位是grid的离散化像素,详见Readme.md解释
        """
        theta_T,T_R = torch.meshgrid([theta_T,T_R]);theta_R,R_R = torch.meshgrid([theta_R,R_R])
        pos_T = torch.polar(T_R,theta_T).view(-1);pos_R = torch.polar(R_R,theta_R).view(-1)#实部为x...展平
        n_receiver,n_transmitter = pos_R.shape[0],pos_T.shape[0]
        self.E_R = torch.empty(n_receiver,n_transmitter,dtype=torch.complex64)#(360,18)
        self.E_inc = torch.empty(Field.nx*Field.ny*4,n_transmitter,dtype=torch.complex64)
        self.theta_T = theta_T;self.theta_R = theta_R;self.T_R = T_R;self.R_R = R_R;
        self.pos_T = pos_T;self.pos_R = pos_R
        self.n_receiver = n_receiver;self.n_transmitter = n_transmitter

        NX, NY = torch.meshgrid([torch.arange(-Field.nx,Field.nx),torch.arange(-Field.ny,Field.ny)])
        pos_N = (NX + 1j*NY)*Field.m_unit#转换单位到m

        self.pos_N = pos_N

        k = omega#*torch.sqrt()#
        for i in trange(n_transmitter,desc="creating incident field"):
            R = torch.abs(pos_T[i]-pos_R)#transmitter与receiver之间距离
            self.E_R[:,i] = 1j/4*hankel_0_1(k*R)#在这里我们改成k*R似乎传入负值计算出来是nan+nanj
            R = torch.abs(pos_T[i]-pos_N).view(-1)#行优先展平，transmitter与each unit之间距离
            self.E_inc[:,i] = 1j/4*hankel_0_1(k*R)

    def set_phi():
        
        

    def get_scatter():
        """
            根据FDFD计算公式,E^s = \phi * vJ
            需要分别计算Phi, vJ = \chi e^inc
        """

        pass


    
if __name__=="__main__":
    from config import R_receiver,R_transmitter
    n_T , n_R = 18,360
    fre = 2#2GHZ
    lambda_ = C_0/(fre*1e9)
    # wvlen = lambda_/m_unit
    omega = 2*torch.pi*C_norm/lambda_
    T_R = 1.2;R_R=1.0#单位m
    theta_T = torch.linspace(0,2*pi,n_T+1)[:-1]
    theta_R = torch.linspace(0,2*pi,n_R+1)[:-1]
    field = Field()
    field.set_incident_E(omega,theta_T,theta_R,T_R,R_R)
    # pos_T = torch.concat([torch.cos(theta_T),torch.sin(theta_T)],dim=1)*R_transmitter
    # pos_R = torch.concat([torch.cos(theta_R),torch.sin(theta_R)],dim=1)*R_receiver
    # E_R = Field.get_ER(omega,theta_T,theta_R,T_R,R_R)
    # E_inc = Field.get_Einc(omega,theta_T,T_R)
    print(field.E_inc.shape)


        


        