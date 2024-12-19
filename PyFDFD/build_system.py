from base import PML,EquationType,FT,GT

def build_system(grid_type,pml,Lpml):
    """
        function [osc, grid3d, s_factor_cell, eps_cell, mu_cell, J_cell, M_cell, Ms, ...
    obj_array, src_array, mat_array, eps_node, mu_node, isiso] = csi_build_system(varargin)
    
    return :
            A矩阵
            M_S矩阵
    """
    ge = grid_type
    pml = pml
    [lprim, Npml] = generate_lprim3d(domain, Lpml, [shape_array, sshape_array], src_array, withuniformgrid);
    grid3d = Grid3d(osc.unit, lprim, Npml, bc);