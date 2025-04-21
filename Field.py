from functions import hankel_0_1,hankel_1_1
from build_object import build_circle
import matplotlib.pyplot as plt
from typing import Tuple,Union,List
from tqdm import trange
from PyFDFD.source import PlaneSrc,Source
from PyFDFD.base import PhysUnit,Oscillation
from PyFDFD.grid import Grid3d
from PyFDFD.base import GT,Axis
from config import *
import torch
import math
import os
# from PIL import Image
# import pandas as pd
# data = pd.read_csv('./data.csv',header=None).values
# data = torch.from_numpy(data)

class Field:
    m_unit = m_unit
    invdom = invdom#(mm)
    # xlchi  = round()
    nx     = int(xs//m_unit)*2#反演分辨率
    ny     = int(ys//m_unit)*2
    Lpml   = [d_pml,d_pml,0]

    def __init__(self) :
        """
        定义离散场类，存储多种离散场,
        receiver记录的E_mea,mesh_grid上的
        E_inc,E_scat,E_tot

        args:
            
        """
        # theta_T = torch.linspace(0,2*pi,n_T+1)[:-1]#transmitter的角度tensor
        # theta_R = torch.linspace(0,2*pi,n_R+1)[:-1]#receiver的角度tensor
        # tmp_domain_x = torch.linspace(-xs,xs,Field.nx)
        # tmp_domain_y = torch.linspace(-ys,ys,Field.ny)
        # step_size = 2*xs/(Field.n的x-1)
        # self.set_incident_E_MOM()
        
    def set_incident_E_FDFD(self,A,b):
        """
        根据AE = B计算电场
        """
        E = torch.linalg.solve(A,b)
        self.E_inc = E
        # return torch.linalg.solve(A,b)
    def set_total_E_FDFD(self,A,b):
        """
        根据AE = B计算电场
        """
        if self.epsil is None:
            raise RuntimeError("Please run Field.set_chi() before using this function!")
        E = torch.linalg.solve(A,b)
        self.E_tot = E
        self.E_scat = E-self.E_inc
    
    def assign_source(self,grid3d:Grid3d,SRCJ_:List[Source],GT_:GT):
        """
        使用FDFD计算入射场和总场，需要设置一个更大包含transmitter,receiver的计算域domain
        """
        for src in SRCJ_:
            src.set_gridtype(GT_)
        """
            elseif ischar(arg) && strcmpi(arg,'SRCJ')
			% Set up sources.
			iarg = iarg + 1; arg = varargin{iarg};
			if ~istypesizeof(arg, 'Source', [1 0])
				warning('Maxwell:buildSys', 'no source is given.');
			end

			while istypesizeof(arg, 'Source', [1 0])
				if istypesizeof(arg, 'TFSFPlaneSrc')
					isTFSF = true;
				end
				srcj_array_curr = arg;
				for src = srcj_array_curr
					src.set_gridtype(ge);
				end
				srcj_array = [srcj_array(1:end), srcj_array_curr];
				iarg = iarg + 1; arg = varargin{iarg};
			end
			iarg = iarg - 1;
        """
        J_cell = [None]*Axis.count()
        M_cell = [None]*Axis.count()
        for w in Axis.elems():
            J_cell[w] = torch.zeros(torch.Size(grid3d.N),dtype = torch.complex32)
            M_cell[w] = torch.zeros(torch.Size(grid3d.N),dtype = torch.complex32)
        for src in SRCJ_:
            for w in Axis.elems():
                ind, JMw_patch = src.generate(w,grid3d)
                if JMw_patch is not None:
                    J_cell[w][tuple(ind)] += JMw_patch
        withdraw = True
        if withdraw:
            if len(SRCJ_)>0:
                import matplotlib.pyplot as plt
                J_array = J_cell[SRCJ_[0].polarization].real.squeeze().numpy()
                plt.imshow(J_array)
                plt.colorbar()
                plt.show()
        else:
            pass
        """
        function [E, H] = solve_eq_direct(eqtype, pml, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d)

        eq = MatrixEquation(eqtype, pml, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d);
        [A, b] = eq.matrix_op();
        """

        return [J_cell_inner.squeeze() for J_cell_inner in J_cell],[M_cell_inner.squeeze() for M_cell_inner in M_cell]

        
    
    def set_incident_E_MOM(self) ->torch.Tensor:
        """
        对于给定的Transmitter的极坐标位置
        计算对于指定receiver位置和unit的入射场
        """

        # assert isinstance(theta_T,torch.Tensor),TypeError("pos_T should be torch.Tensor!")
        # assert isinstance(theta_R,torch.Tensor),TypeError("pos_R should be torch.Tensor!")
        if isinstance(R_transmitter,Union[int,float]):
            T_R = torch.tensor(R_transmitter,dtype=torch.float32)
        if isinstance(R_receiver,Union[int,float]):
            R_R = torch.tensor(R_receiver,dtype=torch.float32)
        # assert pos_T.shape[1]==2,RuntimeError("pos_T should have 2 columns!")
        # assert pos_R.shape[1]==2,RuntimeError("pos_R should have 2 columns!")
        """
            # 在参考CC-CSI代码发现,计算phi的时候,R的计算单位是grid的离散化像素,详见Readme.md解释
            MOM method
        """
        # theta_T = torch.linspace(0,2*pi,n_T+1)[:-1]#transmitter的角度tensor
        # theta_R = torch.linspace(0,2*pi,n_R+1)[:-1]#receiver的角度tensor
        tmp_domain_x = torch.linspace(-xs,xs,Field.nx)
        tmp_domain_y = torch.linspace(-ys,ys,Field.ny)
        self.step_size = 2*xs/(Field.nx-1)#MOM step size
        self.cell_area = self.step_size**2
        self.a_eqv = math.sqrt(self.cell_area/pi)
        # E_s = torch.zeros(n_R,n_T,dtype=torch.complex64)
        self.x_dom,self.y_dom = torch.meshgrid([tmp_domain_x,-tmp_domain_y])
        self.N_cell_dom = Field.nx*Field.ny
        """transmitter的角度，半径和坐标"""
        self.theta_T,self.T_R = torch.meshgrid([theta_T,T_R])
        self.pos_T = torch.polar(self.T_R,self.theta_T)
        pos_T_x = self.pos_T.real.flatten().unsqueeze(0);pos_T_y = self.pos_T.imag.flatten().unsqueeze(0)
        """receiver的角度，半径"""
        self.theta_R,self.R_R = torch.meshgrid([theta_R,R_R])
        self.pos_R = torch.polar(self.R_R,self.theta_R)
        """计算入射场"""
        x0 = self.x_dom.flatten().unsqueeze(1);y0 = self.y_dom.flatten().unsqueeze(1)
        rho_mat_t = torch.sqrt((x0-pos_T_x)**2+(y0-pos_T_y)**2)
        T_mat = 1j/4*1j*k_0*eta_0*hankel_0_1(k_0*rho_mat_t)
        print('T_mat.shape:{}'.format(T_mat.shape))
        self.E_inc = T_mat
        # xi_all = -1j*omega*()
        # theta_T,T_R = torch.meshgrid([theta_T,T_R]);theta_R,R_R = torch.meshgrid([theta_R,R_R])
        # pos_T = torch.polar(T_R,theta_T).view(-1);pos_R = torch.polar(R_R,theta_R).view(-1)#实部为x...展平
        # n_receiver,n_transmitter = pos_R.shape[0],pos_T.shape[0]
        # self.E_R = torch.empty(n_receiver,n_transmitter,dtype=torch.complex64)#(360,18)
        # self.E_inc = torch.empty(Field.nx*Field.ny*4,n_transmitter,dtype=torch.complex64)
        # self.theta_T = theta_T;self.theta_R = theta_R;self.T_R = T_R;self.R_R = R_R;
        # self.pos_T = pos_T;self.pos_R = pos_R
        # self.n_receiver = n_receiver;self.n_transmitter = n_transmitter
        # NX, NY = torch.meshgrid([torch.arange(-Field.nx,Field.nx),torch.arange(-Field.ny,Field.ny)])
        # pos_N = (NX + 1j*NY)*Field.m_unit#转换单位到m

        # self.pos_N = pos_N#单位m

        # k = omega#*torch.sqrt()#
        # self.k = k
        # self.omega = omega
        # for i in trange(n_transmitter,desc="creating incident field"):
        #     R = torch.abs(pos_T[i]-pos_R)#transmitter与receiver之间距离
        #     self.E_R[:,i] = 1j/4*hankel_0_1(k*R)#在这里我们改成k*R似乎传入负值计算出来是nan+nanj
        #     R = torch.abs(pos_T[i]-pos_N).view(-1)#行优先展平，transmitter与each unit之间距离
        #     self.E_inc[:,i] = 1j/4*hankel_0_1(k*R)

    def set_phi(self):

        # assert hasattr(self,'n_receiver'),RuntimeError("Please run Field.set_incident_E() before using this function!")
        # Phi = torch.empty(self.n_receiver,Field.nx*Field.ny*4)
        # R = self.pos_R.view(-1,1)-self.pos_N.view(-1)#行优先展开，receiver与each unit的距离
        # R = torch.abs(R)
        # self.Phi = 1j/4*hankel_0_1(self.k*R)
        # self.Phi/=self.omega
        pass

    # def set_chi(self,omega:float,name):#fre:list
    #     # assert isinstance(fre,list), TypeError('type(fre) should be list not {}!'.format(type(fre)))
    #     assert omega==self.omega, RuntimeError("omega of chi is not equal to Field.omega")
    #     self.chi = Field.get_chi(omega,name,**{'epsilon_robj':3,'sigma_obj':0,'r':0.01,'x_c':0,'y_c':0})
    def set_scatter_E_Born_Appro(self,A):
        """
        FDFD方程：
        (\nabla \times \mu_0^-1 \nalba \times -\omega^2\epsilon_b) E^scat = \omega^2\chi E_p
        使用一阶近似，将入射场代入右侧E_p
        """
        assert self.E_inc is not None, RuntimeError("Please run Field.set_incident_E() before using this function!")
        assert self.epsil is not None, RuntimeError("Please run Field.set_chi() before using this function!")
        chi = self.epsil.flatten().unsqueeze(1)-1
        _,wvlen = Field.get_lambda(fre)#离散波长
        unit = PhysUnit(m_unit)
        osc = Oscillation(wvlen,unit)
        omega = osc.in_omega0()
        b = omega**2*chi*self.E_inc
        b = b.to(torch.complex64)
        self.E_scat = torch.linalg.solve(A,b)
    
        


    def set_scatter_E_MOM(self,omega):
        """
            args:
                omega : 角频率,连续空间中的(与m的单位一致)
        """
        # for i in range(len(self.fre)):
        xi_all = -1j*omega*(self.epsil-1)*eps0*self.cell_area
        xi_all = xi_all.to(torch.complex64)
        # bool_eps = (self.epsil != 1)
        # plt.imshow(bool_eps)
        # plt.colorbar()
        # plt.show()
        # x0 = self.x_dom[bool_eps];y0 = self.y_dom[bool_eps]
        x0 = self.x_dom;y0 = self.y_dom
        x0 = x0.flatten();y0 = y0.flatten()
        # xi_forward = xi_all[bool_eps]
        xi_forward = xi_all
        xi_forward = xi_forward.flatten()
        N_cell = x0.shape[0]
        # Phi_mat = torch.zeros((self.N_cell_dom,self.N_cell_dom),dtype=torch.complex64)
        dist_cell = torch.sqrt((x0.view(-1,1)-x0)**2+(y0.view(-1,1)-y0)**2)
        dist_cell += torch.eye(N_cell)
        I1 = 1j/4*hankel_0_1(k_0*dist_cell)
        Phi_mat = I1*Coef
        Phi_mat = Phi_mat * (torch.ones(N_cell,N_cell)-torch.eye(N_cell))
        I2 = 1j/4*(2/(k_0*self.a_eqv)*hankel_1_1(k_0*torch.tensor(self.a_eqv))+4*1j/(self.cell_area*k_0**2))
        S1 = I2*Coef
        Phi_mat = Phi_mat + S1*torch.eye(N_cell)

        A = torch.eye(N_cell,dtype = torch.complex64)-Phi_mat@torch.diag(xi_forward)
        # E_tot = torch.linalg.solve(A,self.E_inc[bool_eps.flatten(),:])#(N_cell,N_rec)
        E_tot = torch.linalg.solve(A,self.E_inc)#(N_cell,N_rec)
        
        """receiver 坐标"""
        pos_R_x = self.pos_R.real.flatten().unsqueeze(0);pos_R_y = self.pos_R.imag.flatten().unsqueeze(0)
        rho_mat = torch.sqrt((x0.view(-1,1)-pos_R_x)**2+(y0.view(-1,1)-pos_R_y)**2).T
        R_mat = Coef*1j/4*hankel_0_1(k_0*rho_mat)
        E_CDM = R_mat@torch.diag(xi_forward)@E_tot
        self.Es = E_CDM
        self.Phi_mat = Phi_mat
        self.R_mat = R_mat

        # Es_npy = self.Es.numpy()
        # import numpy as np
        # np.save(os.path.join(pwd,'Data','multi_circles','1_Es.npy'),Es_npy)
        # print('Es.shape:{}'.format(self.Es.shape))
        # x0[]
        # assert hasattr(self,'E_inc'), RuntimeError("Please run Field.set_incident_E() before using this function!")
        # self.vJ = self.chi*self.E_inc    #contrast source (N_unit,N_trans)
        # self.E_scat = self.Phi@self.vJ#(N_rece,N_unit) x (N_unit,N_trans)
        # self.E_tot = self.E_R+self.E_scat
        # print(self.E_scat.shape)

    def set_chi(self,load_from_gt:bool = True,**kargs):
        # assert isinstance(fre,list), TypeError('type(fre) should be list not {}!'.format(type(fre)))
        # len_fre = len(fre)
        # chi = torch.empty(Field.nx*Field.ny*4,1,dtype = torch.complex64)#len_fre
        # fre = torch.tensor(fre)
        
        if load_from_gt:
            import numpy as np
            epsil = np.load(**kargs)
            # plt.ion()
            # plt.imshow(chi)
            # plt.colorbar()
            # plt.show()
            # array = (chi/chi.max() * 255).astype(np.uint8)

            # # 使用 PIL 保存，确保图像大小精确为 n×n
            # image = Image.fromarray(array)
            # image.save("1.png")  
            self.epsil = torch.from_numpy(epsil)
        else:
            """
            从self.guess中获取
            """
            epsil = kargs['guess'].epsil
            self.epsil = epsil

        # return chi

    def get_lambda(fre:Union[int,float]):
        """
            根据输入频率得到波长和离散波长
        """
        lambda_ = C_0/(fre*1e9)
        return lambda_,lambda_/Field.m_unit


    def save_Es(self,path):
        """
            根据FDFD计算公式,E^s = \phi * vJ
            需要分别计算Phi, vJ = \chi e^inc
        """
        Es_npy = self.Es.numpy()
        import numpy as np
        np.save(os.path.join(pwd,path,'Es.npy'),Es_npy)
        # print('Es.shape:{}'.format(self.Es.shape))
        # pass


    
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
    field.set_chi(omega,'circle')
    field.set_scatter_E()
    # pos_T = torch.concat([torch.cos(theta_T),torch.sin(theta_T)],dim=1)*R_transmitter
    # pos_R = torch.concat([torch.cos(theta_R),torch.sin(theta_R)],dim=1)*R_receiver
    # E_R = Field.get_ER(omega,theta_T,theta_R,T_R,R_R)
    # E_inc = Field.get_Einc(omega,theta_T,T_R)
    print(field.Phi.shape)
    print(field.E_inc.shape)


        


        