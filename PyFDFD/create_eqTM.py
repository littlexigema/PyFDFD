# import numpy as np
# from base.Axis import Axis
# from base.PML import PML

# def create_eqTM(eqtype,pml,omega,eps_ell,mu_cell,s_factor_cell,J_cell,M_cell,grid3d):
#     N = grid3d.N
#     src_n,_ = J_cell.shape
#     if src_n == 0:
#         src_n = 1
#     r = reordering_indices(Axis.count(), N)
#     # Construct curls
#     dl_factor_cell = [];
#     if pml == PML.sc:
#         dl_factor_cell = s_factor_cell

#     ge = eqtype.ge
#     [Ce, Cm] = create_curls(ge, dl_factor_cell, grid3d)

# def reordering_indices(dof, N):
#     """
#     Generate indices to reorder the elements of matrices and vectors to reduce the
#     bandwidth of the Maxwell operator matrix.
    
#     Parameters:
#     dof (int): Degrees of freedom, should be a positive integer.
#     N (list or np.ndarray): Row vector with integer elements.
    
#     Returns:
#     np.ndarray: Reordered indices.
#     """
#     # Check arguments
#     if not isinstance(dof, int) or dof <= 0:
#         raise ValueError('"dof" should be positive integer.')
#     if not isinstance(N, (list, np.ndarray)) or not all(isinstance(n, int) for n in N):
#         raise ValueError('"N" should be row vector with integer elements.')
    
#     # Generate indices
#     r = np.arange(1, dof * np.prod(N) + 1)
#     r = r.reshape(np.prod(N), dof)
#     r = r.T
#     r = r.flatten()
    
#     return r

# # Example usage
# dof = 3
# N = [4, 5, 6]
# indices = reordering_indices(dof, N)
# print(indices)