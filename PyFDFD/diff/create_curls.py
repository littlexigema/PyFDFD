from ..base import Sign,GT,Axis
# from . import create_Ds,create_masks
from .create_Ds import create_Ds
from .create_masks import create_masks
from torch import sparse
import torch


def create_curls(ge, dl_factor_cell, grid3d):
    # chkarg(isinstance(ge, GT), '"ge" should be instance of GT.')
    # chkarg(dl_factor_cell is None or (isinstance(dl_factor_cell, list) and all(isinstance(cell, torch.Tensor) for cell in dl_factor_cell)), 
    #        '"dl_factor_cell" should be empty, or %d-by-%d cell array whose each element is row vector with real elements.' % (Axis.count(), GT.count()))
    # chkarg(isinstance(grid3d, Grid3d), '"grid3d" should be instance of Grid3d.')

    # Create Df and Db
    Df = create_Ds(Sign.P, ge, dl_factor_cell, grid3d)  # GT.elems(Sign.p) GT.dual, forward
    Db = create_Ds(Sign.N, ge, dl_factor_cell, grid3d)  # GT.elems(Sign.n) GT.prim, backward

    # Create mask matrices
    # ind_Mp, ind_Md = create_masks(ge, grid3d)#目前ind_Mp和ind_Md全部内容为False

    # Form curl matrices
    M = torch.prod(torch.tensor(grid3d.N)).item()
    Z = sparse.FloatTensor(M, M)

    Cp = torch.cat([#ind_Mp和Df,Db的展开方式需要保持一致
        torch.cat([Z, -Df[Axis.Z], Df[Axis.Y]], dim=1),
        torch.cat([Df[Axis.Z], Z, -Df[Axis.X]], dim=1),
        torch.cat([-Df[Axis.Y], Df[Axis.X], Z], dim=1)
    ], dim=0)
    # Cp[:, ind_Mp] = 0
    # Cp[ind_Md, :] = 0

    Cd = torch.cat([
        torch.cat([Z, -Db[Axis.Z], Db[Axis.Y]], dim=1),
        torch.cat([Db[Axis.Z], Z, -Db[Axis.X]], dim=1),
        torch.cat([-Db[Axis.Y], Db[Axis.X], Z], dim=1)
    ], dim=0)
    # Cd[:, ind_Md] = 0
    # Cd[ind_Mp, :] = 0

    if ge == GT.PRIM:
        Ce = Cp  # forward
        Cm = Cd  # backward
    else:  # ge == GT.dual
        Ce = Cd
        Cm = Cp

    return Ce, Cm

# Example usage
# Assuming GT, Axis, Sign, Grid3d, create_Ds, and create_masks are defined elsewhere
# ge = GT.prim  # or GT.dual
# dl_factor_cell = [...]  # Define this based on your requirements
# grid3d = Grid3d(...)  # Define this based on your requirements
# Ce, Cm = create_curls(ge, dl_factor_cell, grid3d)