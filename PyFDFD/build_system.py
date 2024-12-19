from base.PML import PML

def build_system(SolverOpts:object):
    """
        function [osc, grid3d, s_factor_cell, eps_cell, mu_cell, J_cell, M_cell, Ms, ...
    obj_array, src_array, mat_array, eps_node, mu_node, isiso] = csi_build_system(varargin)
    
    return :
            A矩阵
            M_S矩阵
    """
    SolverOpts.pml = PML.sc

    [A, b, ~, ~, ~] = mycreate_eqTM(solveropts.eqtype, solveropts.pml, omega, Eps, mu, s_factor, J, M, grid3d)
