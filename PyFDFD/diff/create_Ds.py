import torch
from torch import sparse
from base.Sign import Sign
from base.Axis import Axis
from base.GT import GT
from base.BC import BC
# from base.grid import Grid2d
from grid.Grid3d import Grid3d

def create_Ds(s, ge, dl_factor_cell, gridnd):
    # Check arguments
    # chkarg(isinstance(s, Sign), '"s" should be instance of Sign.')
    # chkarg(isinstance(ge, GT), '"ge" should be instance of GT.')
    # chkarg(isinstance(gridnd, (Grid2d, Grid3d)), '"gridnd" should be instance of Grid2d or Grid3d.')

    v = Axis.x
    # if isinstance(gridnd, Grid2d):#实际运行中是grid3d
        # v = Dir.h
    chkarg(dl_factor_cell is None or (isinstance(dl_factor_cell, list) and all(isinstance(cell, torch.Tensor) for cell in dl_factor_cell)), 
           '"dl_factor_cell" should be empty, or %d-by-%d cell array whose each element is row vector with real elements.' % (v.count(), GT.count()))

    # Get the shape
    N = gridnd.N

    # Get the relevant derivative matrices
    g = GT.elems(s)
    bc = gridnd.bc

    # Basic setup of Df and Db
    Ds_cell = [None] * v.count()
    if s == Sign.p:  # Ds == Df
        for w in v.elems():
            f1 = 1

            if bc[w] == BC.P:
                fg = torch.exp(-1j * gridnd.kBloch[w] * gridnd.L[w])
            else:  # ghost point BC is BC.e if ge == GT.prim; BC.m if ge == GT.dual
                fg = 0

            Ds_cell[w] = create_Dw(w, N, f1, fg)
    else:  # Ds == Db
        for w in v.elems():
            if (ge == GT.PRIM and bc[w] == BC.M) or (ge == GT.DUAL and bc[w] == BC.E):
                f1 = 2  # symmetry of operator for this case is not implemented yet
            else:
                f1 = 1

            if bc[w] == BC.P:
                fg = torch.exp(-1j * gridnd.kBloch[w] * gridnd.L[w])
            else:  # bc[w] == BC.e or BC.m
                fg = 0  # f1 = 1 or 2 takes care of the ghost point

            Ds_cell[w] = create_Dw(w, N, f1, fg)
            Ds_cell[w] = -Ds_cell[w].conj().t()  # conjugate transpose rather than transpose (hence nonsymmetry for kBloch ~= 0)
            #line 54检查

    dl = [None] * v.count()
    if dl_factor_cell is None:
        dl = torch.meshgrid(*gridnd.dl[:, g])
    else:
        dl_cell = mult_vec(dl_factor_cell[:, g], gridnd.dl[:, g])
        dl = torch.meshgrid(*dl_cell)

    for w in v.elems():
        Ds_cell[w] = create_spdiag(dl[w]**-1) @ Ds_cell[w]

    return Ds_cell

def chkarg(condition, message):
    if not condition:
        raise ValueError(message)

def create_Dw(w, N, f1, fg):
    # This function should create the derivative matrix Dw based on the provided parameters.
    # The implementation of this function is not provided in the original MATLAB code.
    # You need to implement this function based on your specific requirements.
    pass

def create_spdiag(diag_elements):
    # This function creates a sparse diagonal matrix from the provided diagonal elements.
    return sparse.diags(diag_elements)

def mult_vec(dl_factor_cell, dl):
    # This function multiplies the elements of dl_factor_cell with dl.
    # The implementation of this function is not provided in the original MATLAB code.
    # You need to implement this function based on your specific requirements.
    pass

# Example usage
# Assuming GT, Axis, Sign, Grid2d, Grid3d, create_Dw, and create_spdiag are defined elsewhere
# s = Sign.p  # or Sign.n
# ge = GT.prim  # or GT.dual
# dl_factor_cell = [...]  # Define this based on your requirements
# gridnd = Grid3d(...)  # or Grid2d, define this based on your requirements
# Ds_cell = create_Ds(s, ge, dl_factor_cell, gridnd)