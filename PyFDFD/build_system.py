from PyFDFD.grid.generate_lprim3d import generate_lprim3d
from PyFDFD.base import PML,EquationType,FT,GT
from PyFDFD.base.Oscillation import Oscillation
from PyFDFD.grid.Grid3d import Grid3d
from PyFDFD.base.PhysUnit import PhysUnit
from PyFDFD.shape import Box
from PyFDFD.base.BC import BC
from config import *
import numpy as np

def build_system(m_unit,wvlen,grid_type,pml,domain,Lpml):
    """
        function [osc, grid3d, s_factor_cell, eps_cell, mu_cell, J_cell, M_cell, Ms, ...
    obj_array, src_array, mat_array, eps_node, mu_node, isiso] = csi_build_system(varargin)
    
    return :
            A矩阵
            M_S矩阵
    """
    ge = grid_type
    pml = pml
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
    src_array = list()
    withuniformgrid = True
    isepsgiven = False
    [lprim, Npml] = generate_lprim3d(domain, Lpml, list(), src_array, withuniformgrid)

    unit = PhysUnit(m_unit)
    osc = Oscillation(wvlen,unit)

    grid3d = Grid3d(osc.unit, lprim, Npml, BC.P)
    if not isepsgiven:
        eps_node_cell, mu_node_cell = assign_material_node(grid3d)
        eps_cell = np.ones(grid3d.lall[GT.PRIM])
    

    return M_s, A, b

# def generate_lprim3d():
#     pass


if __name__=="__main__":
    pass