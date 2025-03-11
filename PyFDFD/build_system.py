from PyFDFD.material import assign_material_node,Material,mean_material_node
from PyFDFD.base import EquationType,GT,FT,Oscillation,PhysUnit,Axis,PML,BC
from PyFDFD.diff import create_curls,create_Ds,create_Dw,create_masks
from PyFDFD.io import generate_s_factor,EMObject,expand_node_array
from PyFDFD.grid import generate_lprim3d,Grid3d
from PyFDFD.shape import Box
from config import *
import torch
import math
# import numpy as np

def create_eqTM(eqtype,pml,omega,eps_cell,mu_cell,s_factor_cell,J_cell,M_cell,grid3d):
    N = grid3d.N
    if J_cell==None:
        src_n = 0
    elif isinstance(J_cell, torch.Tensor):
        src_n,_ = J_cell.shape
    else:
        raise RuntimeError("J_cell should be a torch.Tensor or None.")
    if src_n == 0:
        src_n = 1
    r = reordering_indices(Axis.count(), N)#后面做测试用
    # Construct curls
    dl_factor_cell = None
    if pml == PML.SC:
        dl_factor_cell = s_factor_cell

    ge = eqtype.ge
    Ce, Cm = create_curls(ge, dl_factor_cell, grid3d)
    
    if pml == PML.U:
        pass
    

def reordering_indices(dof:int, N:int):
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

def build_system(m_unit,wvlen,domain,Lpml,emobj:EMObject):
    """
        function [osc, grid3d, s_factor_cell, eps_cell, mu_cell, J_cell, M_cell, Ms, ...
    obj_array, src_array, mat_array, eps_node, mu_node, isiso] = csi_build_system(varargin)
    
    return :
            A矩阵
            M_S矩阵
    """
    # ge = grid_type
    # pml = pml
    # xlchi           = round(invdom(1) / m_unit)
    # xhchi           = round(invdom(2) / m_unit)
    # ylchi           = round(invdom(3) / m_unit)
    # yhchi           = round(invdom(4) / m_unit)
    # zlchi           = round(invdom(5) / m_unit)
    # zhchi           = round(invdom(6) / m_unit)
    # np_domain       = np.array([[xlchi,xhchi],[ylchi,yhchi],[zlchi,zhchi]])
    # dl              = 1
    # domain  = Box(np_domain, dl)#box domain

    """
        creat instance shape
        shape_array
        sshape_array
    """

    shape_array,sshape_array = None,None#没有实例化的shape
    srcj_array, srcm_array = list(), list()
    src_array = [*srcj_array, *srcm_array]
    withuniformgrid = True
    isepsgiven = False
    """
    solveropts结构体成员
    """
    eqtype = EquationType(FT.E,GT.PRIM)#solveropts.eqtype = ...
    pml = PML.SC #solveropts.pml = ...
    [lprim, Npml] = generate_lprim3d(domain, Lpml, list(), src_array, withuniformgrid)

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
    J_cell, M_cell, Ms = [None]*3
    if TME_mode == "TM":
        A,b = create_eqTM(eqtype,pml,omega,eps_cell,mu_cell,s_factor_cell,J_cell,M_cell,grid3d)
    return M_s, A, b

def myassign_source(grid3d:Grid3d, srcj_array, srcm_array):
    """
    function [J_cell, M_cell, Ms] = myassign_source(grid3d, srcj_array, srcm_array)
    暂时不实现这个函数的功能，似乎A矩阵的构建不需要这个
    """
    if not isinstance(grid3d,Grid3d):
        raise ValueError('"grid3d" should be an instance of Grid3d.')
    if not isinstance(srcj_array,list) or not isinstance(e,):
        raise ValueError('"srcj_array" should be a list.')

    J_cell = [None * Axis.count()]
    M_cell = [None * Axis.count()]
    Ms = None
    for w in Axis.elems():
        J_cell[w] = np.zeros(grid3d.lall[GT.NODE])
        M_cell[w] = np.zeros(grid3d.lall[GT.NODE])
    for src in srcj_array:
        src.assign_source(grid3d, J_cell)
    for src in srcm_array:
        src.assign_source(grid3d, M_cell)
    return J_cell, M_cell, Ms


# def generate_lprim3d():
#     pass


if __name__=="__main__":
    pass
