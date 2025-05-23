from PyFDFD.material import assign_material_node,Material,mean_material_node,assign_material_node_any_shape
from PyFDFD.base import EquationType,GT,FT,Oscillation,PhysUnit,Axis,PML,BC
from PyFDFD.diff import create_curls,create_Ds,create_Dw,create_masks
from PyFDFD.io import generate_s_factor,EMObject,expand_node_array
from PyFDFD.grid import generate_lprim3d,Grid3d
import matplotlib.pyplot as plt
from PyFDFD.shape import Box
from config import *
import torch
import math
import numpy as np

class Forward_basic:
    def __init__(self):
        pass
    def create_eqTM(self,eqtype,pml,omega,eps_cell,mu_cell,s_factor_cell,J_cell,M_cell,grid3d):
        """由于电流源的不确定性，我们并不直接计算J"""
        N = grid3d.N
        # if J_cell==None:
        #     src_n = 0
        # elif isinstance(J_cell, torch.Tensor):
        #     src_n,_ = J_cell.shape
        # else:
        #     pass
        #     # raise RuntimeError("J_cell should be a torch.Tensor or None.")
        # if src_n == 0:
        #     src_n = 1
        r = self.reordering_indices(Axis.count(), N)#后面做测试用
        # Construct curls
        dl_factor_cell = None
        if pml == PML.SC:
            dl_factor_cell = s_factor_cell

        ge = eqtype.ge
        Ce, Cm = create_curls(ge, dl_factor_cell, grid3d)
        mu = torch.concat([mu_inner.flatten() for mu_inner in mu_cell])
        eps = torch.concat([eps_inner.flatten() for eps_inner in eps_cell])
        """特殊值处理"""
        ind_pec = torch.isinf(abs(eps))
        eps[ind_pec] = 1

        pm = torch.ones_like(ind_pec)  # PEC mask
        pm[ind_pec] = 0
        n = pm.size(0)
        index = torch.arange(n).repeat(2,1)
        PM = torch.sparse_coo_tensor(index,pm + 0j,size=(n,n))
        # j,m = None,None
        """
        CC-CSI代码
        j = sparse([]);
        m = sparse([]);
        for ii = 1:src_n
            tmp1 = [J_cell{ii,Axis.x}(:); J_cell{ii,Axis.y}(:); J_cell{ii,Axis.z}(:)];
            tmp2 = [M_cell{ii,Axis.x}(:); M_cell{ii,Axis.y}(:); M_cell{ii,Axis.z}(:)];
            j = [j, sparse(tmp1)];
            m = [m, sparse(tmp2)];
        end
        """
        """
        maxwell-FDFD:
        this.j = [J_cell{Axis.x}(:) ; J_cell{Axis.y}(:) ; J_cell{Axis.z}(:)];
		this.m = [M_cell{Axis.x}(:) ; M_cell{Axis.y}(:) ; M_cell{Axis.z}(:)];
        """
        j = torch.concat([j_inner.flatten() for j_inner in J_cell])
        m = torch.concat([m_inner.flatten() for m_inner in M_cell])
        if eqtype.f == FT.E:
            INV_MU = torch.sparse_coo_tensor(index,1 / mu + 0j,size=(n,n)) #when mu has Inf, "MU \ Mat" complains about singularity
            D = torch.sparse_coo_tensor(index,eps + 0j,size=(n,n))
            
            A_for = PM @ (Cm @ INV_MU @ Ce) @ PM
            A_back = - omega**2 * D
            # hfcn_A = @(e) pm .* (Cm * ((Ce * (pm .* e)) ./ mu)) - omega^2 * (eps .* e);
            # hfcn_Atr = @(e) pm .* (Ce.' * ((Cm.' * (pm .* e)) ./ mu)) - omega^2 * (eps .* e);
            
            b = -1j *omega*j - Cm@(m/mu)
            """
                original code
                INV_MU = create_spdiag(1./this.mu);  % create INV_MU instead of inverting MU; "MU \ Mat" complains about singularity when mu has Inf
				EPS = create_spdiag(this.eps);
				PM = create_spdiag(this.pm);	

				A = PM * (this.Cm * INV_MU * this.Ce) * PM - this.omega^2 * EPS;
				b = -1i*this.omega*this.j - this.Cm*(this.m./this.mu);
            
				A = A(this.r, this.r);
				b = b(this.r);
            """
        elif eqtype.f == FT.H:
            INV_EPS = torch.sparse_coo_tensor(index,1 / eps,size=(n,n))
            D = torch.sparse_coo_tensor(index,mu,size=(n,n))
            A_for = (Ce @ INV_EPS @ Cm)
            A_back = - omega**2 @ D    
            b = -1j * omega * m - Ce @ (j / mu)
        return A_for, A_back,b
            # hfcn_GfromF = @(e) (Ce * e + m) ./ (-1i*omega*repmat(mu,1,src_n));
        # mu = [mu_cell{Axis.x}(:) ; mu_cell{Axis.y}(:) ; mu_cell{Axis.z}(:)];
        # eps = [eps_cell{Axis.x}(:) ; eps_cell{Axis.y}(:) ; eps_cell{Axis.z}(:)];
        # % j = cell(1,src_n); m = cell(1,src_n);
        # j = sparse([]);
        # m = sparse([]);
        # for ii = 1:src_n
        #     tmp1 = [J_cell{ii,Axis.x}(:); J_cell{ii,Axis.y}(:); J_cell{ii,Axis.z}(:)];
        #     tmp2 = [M_cell{ii,Axis.x}(:); M_cell{ii,Axis.y}(:); M_cell{ii,Axis.z}(:)];
        #     j = [j, sparse(tmp1)];
        #     m = [m, sparse(tmp2)];
        # end
        # if pml == PML.U:
        #     pass

    def reordering_indices(self,dof:int, N:int):
        """
        Generate indices to reorder the elements of matrices and vectors to reduce the
        bandwidth of the Maxwell operator matrix.
        
        Parameters:
        dof (int): Degrees of freedom, should be a positive integer.
        N (list or np.ndarray): Row vector with integer elements.
        
        Returns:
        np.ndarray: Reordered indices.
        """
        # Check arguments
        if not isinstance(dof, int) or dof <= 0:
            raise ValueError('"dof" should be positive integer.')
        if isinstance(N, torch.Tensor):
            N = N.to(torch.long).tolist()
        elif isinstance(N,list):
            pass
        else:
            raise ValueError('"N" should be row vector with integer elements.')
        # if not isinstance(N, (list, torch.Tensor)) or not all(isinstance(n, int) for n in N):
        #     raise ValueError('"N" should be row vector with integer elements.')
        
        # Generate indices
        r = torch.arange(1, dof * math.prod(N) + 1).resize(dof, math.prod(N))
        """
        r = reshape(r, prod(N), dof);
        r = r.';
        与line67等价
        """
        r = r.T.flatten()
        """
        r.T.flatten()与matlabr = r(:);展开方式等价
        """
        return r

    def build_system(self,m_unit,wvlen,domain,Lpml,emobj:EMObject,srcj_array:list=list(),srcm_array:list=list()):
        """
            function [osc, grid3d, s_factor_cell, eps_cell, mu_cell, J_cell, M_cell, Ms, ...
        obj_array, src_array, mat_array, eps_node, mu_node, isiso] = csi_build_system(varargin)
        
        return :
                A矩阵
        """

        shape_array,sshape_array = None,None#没有实例化的shape
        # srcj_array, srcm_array = list(), list()
        src_array = [*srcj_array, *srcm_array]
        # withuniformgrid = False#动态生成网格
        withuniformgrid = True
        isepsgiven = False
        """
        solveropts结构体成员
        """
        eqtype = EquationType(FT.E,GT.PRIM)#solveropts.eqtype = ...
        pml = PML.SC #solveropts.pml = ...
        for src in src_array:
            src.set_gridtype(eqtype.ge)
        if not self.inverse:
            shape_array = [self.domain_compute]#计算domain,用Box代替
        [lprim, Npml] = generate_lprim3d(domain, Lpml, shape_array, src_array, withuniformgrid)
        if True:
            x = lprim[0]
            y = lprim[1]

            # 创建网格点
            X, Y = torch.meshgrid(x, y, indexing="ij")

            # 绘制网格
            plt.figure(figsize=(8, 8))
            plt.scatter(X.numpy(), Y.numpy(), s=1, color="black")  # 绘制网格点
            R = pos_T_for.real[0].item()
            circle = plt.Circle((0, 0), R, color="red", fill=False, linewidth=2, label="Red Circle")  # 圆心(0, 0)，半径100
            plt.gca().add_artist(circle)
            plt.gca().set_aspect("equal", adjustable="box")  # 设置坐标轴比例相等
            plt.title("2D Grid Visualization")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()
        """获取计算domain x,y mask"""
        # pos_Tx, pos_Ty = pos_T_for.real,pos_T_for.imag
        #self.domain_compute与lprim比较
        # x_mask = torch.where(lprim[0].unsqueeze(1)==pos_Tx)[0]
        # y_mask = torch.where(lprim[1].unsqueeze(1)==pos_Ty)[0]
        # self.x_mask = x_mask
        # self.y_mask = y_mask

        unit = PhysUnit(m_unit)
        osc = Oscillation(wvlen,unit)

        grid3d = Grid3d(osc.unit, lprim, Npml, BC.P)

        #Set up the degree of the polynomial grading of the PML scale factors.
        deg_pml = 4
        #Set up the target reflection coefficient of the PML.
        R_pml = math.exp(-16)
        #目前当作isepsgiven没有给定，false
        if not isepsgiven:
            eps_node_cell, mu_node_cell = assign_material_node(grid3d,emobj,None,None)
            # eps_cell = np.ones(grid3d.lall[GT.PRIM])
        eps_cell = mean_material_node(grid3d,eqtype.ge,eps_node_cell)#不让mean_material_node内部对material_node改变影响外部变量
        mu_cell = mean_material_node(grid3d,eqtype.ge,mu_node_cell)
        #construct PML s-factors.
        s_factor_cell = generate_s_factor(osc.in_omega0(), grid3d, deg_pml, R_pml)
        # eps_node = [None] * Axis.count()
        # mu_node = [None] * Axis.count()
        # for w in Axis.elems():
        #     eps_node_cell[w] = expand_node_array(grid3d,eps_node_cell[w])
        #     mu_node_cell[w] = expand_node_array(grid3d,mu_node_cell[w])
        #     eps_node[w] = eps_node_cell[w]
        #     mu_node[w] = mu_node_cell[w]
        # Construct sources.
        # 暂时注释这行
        # [J_cell, M_cell, Ms] = myassign_source(grid3d, srcj_array, srcm_array)
        # J_cell, M_cell, Ms = [None]*3
        J_cell, M_cell = self.field.assign_source(grid3d, srcj_array,eqtype.ge)
        if TME_mode == "TM":
            A_for,A_back,b = self.create_eqTM(eqtype,pml,osc.in_omega0(),eps_cell,mu_cell,s_factor_cell,J_cell,M_cell,grid3d)
        self.mu_node_cell = mu_node_cell
        self.eps_node_cell = eps_node_cell
        self.s_factor_cell = s_factor_cell
        self.unit = unit
        self.osc = osc
        self.grid3d = grid3d
        self.pml = pml
        return A_for,A_back,b

    def build_system_back(self):
        """
        needs to check
        """
        isepsgiven = False
        eqtype = EquationType(FT.E,GT.PRIM)
        if not isepsgiven:
            assert self.guess.epsil is not None, RuntimeError('Guess should have epsil before use this function:{}'.format("def build_system_back(self)s"))
            eps_node_cell= assign_material_node_any_shape(self.grid3d,self.guess.epsil)
        eps_cell = mean_material_node(self.grid3d,eqtype.ge,eps_node_cell)#不让mean_material_node内部对material_node改变影响外部变量
        mu_cell = mean_material_node(self.grid3d,eqtype.ge,self.mu_node_cell)
        A_back = self.generate_A_back(self.osc.in_omega0(),eps_cell)
        self.eps_node_cell = eps_node_cell
        return A_back

        

    def generate_A_back(self,omega,eps_cell):
        eps = torch.concat([eps_inner.flatten() for eps_inner in eps_cell])
        ind_pec = torch.isinf(abs(eps))
        eps[ind_pec] = 1
        n = ind_pec.size(0)
        index = torch.arange(n).repeat(2,1)
        D = torch.sparse_coo_tensor(index,eps + 0j,size=(n,n)).to(torch.complex64)
        A_back = - omega**2 * D
        return A_back

    def myassign_source(self,grid3d:Grid3d, srcj_array, srcm_array):
        """
        function [J_cell, M_cell, Ms] = myassign_source(grid3d, srcj_array, srcm_array)
        暂时不实现这个函数的功能，似乎A矩阵的构建不需要这个
        """
        # if not isinstance(grid3d,Grid3d):
        #     raise ValueError('"grid3d" should be an instance of Grid3d.')
        # if not isinstance(srcj_array,list) or not isinstance(e,):
        #     raise ValueError('"srcj_array" should be a list.')

        # J_cell = [None * Axis.count()]
        # M_cell = [None * Axis.count()]
        # Ms = None
        # for w in Axis.elems():
        #     J_cell[w] = np.zeros(grid3d.lall[GT.NODE])
        #     M_cell[w] = np.zeros(grid3d.lall[GT.NODE])
        # for src in srcj_array:
        #     src.assign_source(grid3d, J_cell)
        # for src in srcm_array:
        #     src.assign_source(grid3d, M_cell)
        # return J_cell, M_cell, Ms
        pass


# def generate_lprim3d():
#     pass


if __name__=="__main__":
    pass
