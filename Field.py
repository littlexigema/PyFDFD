from build_object import build_circle
from functions import hankel_0_1
from typing import Tuple,Union
from tqdm import trange
from config import *
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

        self.pos_N = pos_N#单位m

        k = omega#*torch.sqrt()#
        self.k = k
        self.omega = omega
        for i in trange(n_transmitter,desc="creating incident field"):
            R = torch.abs(pos_T[i]-pos_R)#transmitter与receiver之间距离
            self.E_R[:,i] = 1j/4*hankel_0_1(k*R)#在这里我们改成k*R似乎传入负值计算出来是nan+nanj
            R = torch.abs(pos_T[i]-pos_N).view(-1)#行优先展平，transmitter与each unit之间距离
            self.E_inc[:,i] = 1j/4*hankel_0_1(k*R)

    def set_phi(self):

        assert hasattr(self,'n_receiver'),RuntimeError("Please run Field.set_incident_E() before using this function!")

        # Phi = torch.empty(self.n_receiver,Field.nx*Field.ny*4)
        R = self.pos_R.view(-1,1)-self.pos_N.view(-1)#行优先展开，receiver与each unit的距离
        R = torch.abs(R)
        self.Phi = 1j/4*hankel_0_1(self.k*R)
        self.Phi/=self.omega

    def set_chi(self,omega:float):#fre:list
        # assert isinstance(fre,list), TypeError('type(fre) should be list not {}!'.format(type(fre)))
        assert omega==self.omega, RuntimeError("omega of chi is not equal to Field.omega")
        self.chi = Field.get_chi(omega,name='circle',**{'epsilon_robj':3,'sigma_obj':0,'r':0.01,'x_c':0,'y_c':0})

    def set_scatter_E(self):
        # for i in range(len(self.fre)):
        assert hasattr(self,'E_inc'), RuntimeError("Please run Field.set_incident_E() before using this function!")
        self.vJ = self.chi*self.E_inc    #contrast source (N_unit,N_trans)
        self.E_scat = self.Phi@self.vJ#(N_rece,N_unit) x (N_unit,N_trans)
        self.E_tot = self.E_R+self.E_scat
        # print(self.E_scat.shape)

    def get_chi(omega,name,**kargs):
        # assert isinstance(fre,list), TypeError('type(fre) should be list not {}!'.format(type(fre)))
        # len_fre = len(fre)
        chi = torch.empty(Field.nx*Field.ny*4,1,dtype = torch.complex64)#len_fre
        # fre = torch.tensor(fre)
        if name=="circle":
            # for i in range(len_fre):
            chi = build_circle(omega,Field.nx,Field.ny,Field.m_unit,Field.m_unit,eps_b,sigma_b,**kargs)
        return chi

    def get_scatter():
        """
            根据FDFD计算公式,E^s = \phi * vJ
            需要分别计算Phi, vJ = \chi e^inc
        """

        pass


    
if __name__=="__main__":
    # from config import R_receiver,R_transmitter
    n_T , n_R = 18,360
    # fre = 2#2GHZ
    lambda_ = C_0/(fre*1e9)
    # wvlen = lambda_/m_unit
    omega = 2*torch.pi*C_norm/lambda_
    T_R = 1.2;R_R=1.0#单位m
    theta_T = torch.linspace(0,2*pi,n_T+1)[:-1]
    theta_R = torch.linspace(0,2*pi,n_R+1)[:-1]
    field = Field()
    field.set_incident_E(omega,theta_T,theta_R,T_R,R_R)
    field.set_phi()
    field.set_chi(fre)
    field.set_scatter_E()
    # pos_T = torch.concat([torch.cos(theta_T),torch.sin(theta_T)],dim=1)*R_transmitter
    # pos_R = torch.concat([torch.cos(theta_R),torch.sin(theta_R)],dim=1)*R_receiver
    # E_R = Field.get_ER(omega,theta_T,theta_R,T_R,R_R)
    # E_inc = Field.get_Einc(omega,theta_T,T_R)
    print(field.Phi.shape)
    print(field.E_inc.shape)


        


        