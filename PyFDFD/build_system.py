from PyFDFD.material.assign_material_node import assign_material_node
from PyFDFD.material.mean_material_node import mean_material_node
from PyFDFD.io.generate_s_factor import generate_s_factor
from PyFDFD.grid.generate_lprim3d import generate_lprim3d
from PyFDFD.base import PML,EquationType,FT,GT
from PyFDFD.material.Material import Material
from PyFDFD.io.EMObject import EMObject
from PyFDFD.base.Oscillation import Oscillation
from PyFDFD.grid.Grid3d import Grid3d
from PyFDFD.base.PhysUnit import PhysUnit
from PyFDFD.io.expand_node_array import expand_node_array
from PyFDFD.base.Axis import Axis
from PyFDFD.base.PML import PML
from PyFDFD.shape import Box
from PyFDFD.base.BC import BC
from config import *
import math
# import numpy as np

def create_eqTM(eqtype,pml,omega,eps_ell,mu_cell,s_factor_cell,J_cell,M_cell,grid3d):
    N = grid3d.N
    src_n,_ = J_cell.shape
    if src_n == 0:
        src_n = 1
    r = reordering_indices(Axis.count(), N)
    # Construct curls
    dl_factor_cell = [];
    if pml == PML.sc:
        dl_factor_cell = s_factor_cell

    ge = eqtype.ge
    [Ce, Cm] = create_curls(ge, dl_factor_cell, grid3d)

def reordering_indices(dof, N):
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
    if not isinstance(N, (list, np.ndarray)) or not all(isinstance(n, int) for n in N):
        raise ValueError('"N" should be row vector with integer elements.')
    
    # Generate indices
    r = np.arange(1, dof * np.prod(N) + 1)
    r = r.reshape(np.prod(N), dof)
    r = r.T
    r = r.flatten()
    
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
    eps_cell = mean_material_node(grid3d,eqtype.ge,eps_node_cell)
    mu_cell = mean_material_node(grid3d,eqtype.ge,mu_node_cell)
    #construct PML s-factors.
    s_factor_cell = generate_s_factor(osc.in_omega0(), grid3d, deg_pml, R_pml)
    eps_node = [None] * Axis.count()
    mu_node = [None] * Axis.count()
    for w in Axis.elems():
        eps_node_cell[w] = expand_node_array(grid3d,eps_node_cell[w])
        mu_node_cell[w] = expand_node_array(grid3d,mu_node_cell[w])
        eps_node[w] = eps_node_cell[w]
        mu_node[w] = mu_node_cell[w]
    # Construct sources.
    # 暂时注释这行
    # [J_cell, M_cell, Ms] = myassign_source(grid3d, srcj_array, srcm_array)
    J_cell, M_cell, Ms = [None]*3
    if TME_mode == "TM":
        A,b = create_eqTM(eqtype,pml,omega,eps_node,mu_node,s_factor_cell,J_cell,M_cell,grid3d)
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