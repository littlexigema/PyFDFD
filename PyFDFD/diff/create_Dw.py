import torch
from torch import sparse
from ..base import Axis
# from base.Dir import Dir

def create_Dw(w, N, f1, fg):
    """
    Creates the forward difference matrix (without dl division). For the backward
    difference matrix, take the transpose or conjugate transpose appropriately, as
    shown in create_Ds().

    f1: factor multiplied to the first element in the w-direction (for implementing the symmetry boundary)
    fg: factor multiplied to the ghost element in the w-direction (for implementing the Bloch boundary)
    """
    # chkarg(isinstance(w, (Axis, Dir)), '"w" should be instance of Axis or Dir.')
    # chkarg(isinstance(N, (list, torch.Tensor)) and len(N) == w.count(), '"N" should be length-%d row vector with integer elements.' % w.count())
    # chkarg(isinstance(f1, complex), '"f1" should be complex.')
    # chkarg(isinstance(fg, complex), '"fg" should be complex.')

    # Translate spatial indices (i,j,k) into matrix indices.
    M = torch.prod(torch.tensor(N))
    row_ind = torch.arange(1, M + 1)
    col_ind_curr = torch.arange(1, M + 1)

    col_ind_next = torch.arange(1, M + 1).reshape(N)
    shift = torch.zeros(w.count(), dtype=torch.int)
    shift[w.value] = -1
    col_ind_next = torch.roll(col_ind_next, shifts=tuple(shift.tolist()), dims=w.value)

    a_curr = torch.ones(N, dtype=torch.complex64)
    a_ind = [slice(None)] * w.count()
    a_ind[w.value] = 0
    a_curr[tuple(a_ind)] = f1

    a_next = torch.ones(N, dtype=torch.complex64)
    a_ind[w.value] = N[w.value] - 1
    a_next[tuple(a_ind)] = fg

    # Create the sparse matrix.
    row_ind = row_ind.repeat(2)
    col_ind = torch.cat([col_ind_curr.flatten(), col_ind_next.flatten()])
    values = torch.cat([-a_curr.flatten(), a_next.flatten()])

    Dw = sparse.FloatTensor(torch.stack([row_ind, col_ind]), values, torch.Size([M, M]))

    return Dw


# Example usage
# Assuming Axis, Dir, and other necessary components are defined elsewhere
# w = Axis.x  # or Dir.h
# N = [10, 10, 10]  # Example grid size
# f1 = 1 + 0j  # Example complex factor
# fg = 1 + 0j  # Example complex factor
# Dw = create_Dw(w, N, f1, fg)