def create_eqTM(eqtype,pml,omega,eps_ell,mu_cell,s_factor_cell,J_cell,M_cell,grid3d):
    N = grid3d.N
    src_n,_ = J_cell.shape
    if src_n == 0:
        src_n = 1
    