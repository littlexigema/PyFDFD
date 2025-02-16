import torch
from ..base import Axis,GT,BC
# from base.Grid2d import Grid2d
from ..grid import Grid3d

def create_masks(ge, gridnd):
    """
    Generating the indices to mask is better than generating a diagonal mask
    matrix, because multiplying the mask matrix cannot make NaN or Inf elements
    zeros.
    """
    # chkarg(isinstance(ge, GT), '"ge" should be instance of GT.')
    # chkarg(isinstance(gridnd, (Grid2d, Grid3d)), '"gridnd" should be instance of Grid2d or Grid3d.')

    v = Axis.x
    # if isinstance(gridnd, Grid2d):#实际运行中是grid3d
    #     v = Dir.h

    # Get the shape
    N = gridnd.N
    bc = gridnd.bc
    ind = [slice(None)] * v.count()

    # Mask matrices
    mask_p = [None] * v.count()
    for w in v.elems():
        mask = torch.zeros(N, dtype=torch.bool)
        us = [u for u in v.elems() if u != w]

        for u in us:
            if (ge == GT.prim and bc[u] == BC.e) or (ge == GT.dual and bc[u] == BC.m):
                ind = [slice(None)] * v.count()
                ind[u] = 0
                mask[tuple(ind)] = True
        mask_p[w] = mask.flatten()

    ind_Mp = torch.cat(mask_p).nonzero().flatten()

    mask_d = [None] * v.count()
    for w in v.elems():
        mask = torch.zeros(N, dtype=torch.bool)

        if (ge == GT.prim and bc[w] == BC.e) or (ge == GT.dual and bc[w] == BC.m):
            ind = [slice(None)] * v.count()
            ind[w] = 0
            mask[tuple(ind)] = True
        mask_d[w] = mask.flatten()

    ind_Md = torch.cat(mask_d).nonzero().flatten()

    return ind_Mp, ind_Md

# def chkarg(condition, message):
#     if not condition:
#         raise ValueError(message)

# Example usage
# Assuming GT, Axis, BC, Grid2d, Grid3d, and other necessary components are defined elsewhere
# ge = GT.prim  # or GT.dual
# gridnd = Grid3d(...)  # or Grid2d, define this based on your requirements
# ind_Mp, ind_Md = create_masks(ge, gridnd)