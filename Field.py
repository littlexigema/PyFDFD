from functions import hankel_0_1
from config import m_unit,R_transmitter,R_receiver,C_0,C_norm
from typing import Tuple
import torch
# import pandas as pd
# data = pd.read_csv('./data.csv',header=None).values
# data = torch.from_numpy(data)


class Field:
    m_unit = m_unit
    def __init__(self,n_t:int,n_r:int,m_unit:float,region_size:Tuple[int,int]) :
        """
        定义离散场类，存储多种离散场,
        receiver记录的E_mea,mesh_grid上的
        E_inc,E_scat,E_tot

        args:
            n_t (int): transmitter的个数
            n_r (int): receiver的个数
            m_unit (float): 每个unit的大小边长(m)
            region_size (tuple): mesh_grid的形状
        """
        self.num_transmitter = n_t
        self.num_receiver = n_r
        self.region_size = region_size
    
    def get_ER(omega,theta_T:torch.tensor,theta_R:torch.tensor,T_R ,R_R ) ->torch.Tensor:
        """
        对于给定的某个Transmitter的位置
        计算对于指定receiver位置的入射场
        
        args:
            omega : 角频率,连续空间中的(与m的单位一致)
            pos_T (torch.Tensor): x,y位置(m)
            pos_R (torch.Tensor): each x,y位置(m)
        """

        assert isinstance(theta_T,torch.Tensor),TypeError("pos_T should be torch.Tensor!")
        assert isinstance(theta_R,torch.Tensor),TypeError("pos_R should be torch.Tensor!")
        if isinstance(T_R,int):
            T_R = torch.tensor(T_R)
        if isinstance(R_R,int):
            R_R = torch.tensor(R_R)
        # assert pos_T.shape[1]==2,RuntimeError("pos_T should have 2 columns!")
        # assert pos_R.shape[1]==2,RuntimeError("pos_R should have 2 columns!")
        """
            在参考CC-CSI代码发现,计算phi的时候,R的计算单位是grid的离散化像素,详见Readme.md解释
        """
        theta_T,T_R = torch.meshgrid([theta_T,T_R]);theta_R,R_R = torch.meshgrid([theta_R,R_R])
        pos_T = torch.polar(T_R,theta_T).view(-1);pos_R = torch.polar(R_R,theta_R).view(-1)#实部为x...展平
        n_receiver,n_transmitter = pos_R.shape[0],pos_T.shape[0]
        E_R = torch.empty(n_receiver,n_transmitter,dtype=torch.complex64)#(360,18)

        k = omega#*torch.sqrt()#
        # T_x,T_y = torch.split(pos_T,[1,1],dim=1)
        # R_x,R_y = torch.split(pos_R,[1,1],dim=1)
        for i in range(n_transmitter):
            # D_x = R_x - T_x[i,:]
            # D_y = R_y - T_y[i,:]
            R = torch.abs(pos_T[i])
            E_R[:,i] = 1j/4*hankel_0_1(k*R).squeeze(1)#在这里我们改成k*R似乎传入负值计算出来是nan+nanj
        # RRc = torch.tensor([])
        # test = hankel_0_1(k*data.T)
        return E_R


    
if __name__=="__main__":
    from config import R_receiver,R_transmitter
    n_T , n_R = 18,360
    fre = 2#2GHZ
    lambda_ = C_0/(fre*1e9)
    # wvlen = lambda_/m_unit
    omega = 2*torch.pi*C_norm/lambda_

    theta_T = [num*(360/n_T)for num in range(n_T)]
    theta_R = [num*(360/n_R)for num in range(n_R)]
    
    theta_T = torch.pi/180*torch.tensor(theta_T).reshape(-1,1)
    theta_R = torch.pi/180*torch.tensor(theta_R).reshape(-1,1)
    pos_T = torch.concat([torch.cos(theta_T),torch.sin(theta_T)],dim=1)*R_transmitter
    pos_R = torch.concat([torch.cos(theta_R),torch.sin(theta_R)],dim=1)*R_receiver
    E_R = Field.get_ER(omega,pos_T,pos_R)
    print(E_R.shape)


        


        