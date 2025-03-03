import torch
from torch import sparse
from ..base import Sign,Axis,GT,BC
# from base.grid import Grid2d
from ..grid import Grid3d
import math

def create_Ds(s, ge, dl_factor_cell, gridnd):
    # Check arguments
    # chkarg(isinstance(s, Sign), '"s" should be instance of Sign.')
    # chkarg(isinstance(ge, GT), '"ge" should be instance of GT.')
    # chkarg(isinstance(gridnd, (Grid2d, Grid3d)), '"gridnd" should be instance of Grid2d or Grid3d.')

    # v = Axis.X
    # if isinstance(gridnd, Grid2d):#实际运行中是grid3d
        # v = Dir.h
    # chkarg(dl_factor_cell is None or (isinstance(dl_factor_cell, list) and all(isinstance(cell, torch.Tensor) for cell in dl_factor_cell)), 
    #        '"dl_factor_cell" should be empty, or %d-by-%d cell array whose each element is row vector with real elements.' % (v.count(), GT.count()))
    #经过先前检查dl_factor_cell与matlab中值一致，所以不再检查
    # Get the shape
    N = gridnd.N

    # Get the relevant derivative matrices
    g = GT.elems(s)
    bc = gridnd.bc

    # Basic setup of Df and Db
    Ds_cell = [None] * Axis.count()
    if s == Sign.P:  # Ds == Df
        for w in Axis.elems():
            f1 = 1

            if bc[w] == BC.P:
                fg = torch.exp(-1j * gridnd.kBloch[w] * gridnd.L[w])
            else:  # ghost point BC is BC.e if ge == GT.prim; BC.m if ge == GT.dual
                fg = 0

            Ds_cell[w] = create_Dw(w, N, f1, fg)
    else:  # Ds == Db
        for w in Axis.elems():
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

    dl = [None] * Axis.count()
    if dl_factor_cell is None:#create_curl dl_factor_cell初始化为None
        dl = torch.meshgrid(*gridnd.dl[:, g])#需要修改
    else:
        # dl_cell = mult_vec(dl_factor_cell[:, g], gridnd.dl[:, g])
        dl_cell = mult_vec(list(list(zip(*dl_factor_cell))[g]),list(list(zip(*gridnd.dl))[g]))
        dl = list(torch.meshgrid(*dl_cell,indexing='ij'))

    for w in Axis.elems():
        # assert 
        Ds_cell[w] = (dl[w].T.flatten().reciprocal()).unsqueeze(1) * Ds_cell[w]
        # Ds_cell[w] = create_spdiag(dl[w]**-1) @ Ds_cell[w]

    return Ds_cell

def chkarg(condition, message):
    if not condition:
        raise ValueError(message)

def create_Dw(w:Axis, N:torch.Tensor, f1, fg):
    # This function should create the derivative matrix Dw based on the provided parameters.
    # The implementation of this function is not provided in the original MATLAB code.
    # You need to implement this function based on your specific requirements.
    """
    在处理稀疏矩阵时较普通方法有优势
    """
    dim = (N != 1).sum().item()
    M = math.prod(N)
    row_ind = torch.arange(M)
    col_ind_curr = torch.arange(M)
    col_ind_next = torch.arange(M).reshape(torch.Size(N)[::-1]).squeeze()
    assert dim ==2, RuntimeError("N shoud be 2D tensor")
    # shift = torch.zeros_like(col_ind_next.shape, dtype=torch.int)
    shifts, dims= -1, w.value^1#w.value^1异或逻辑，同或int(not(bool(w.value)^bool(0)))#w.value

    a_curr = torch.ones(torch.Size(N)[::-1]).squeeze()
    a_ind_curr = [slice(None)]*dim
    a_next = torch.ones(torch.Size(N)[::-1]).squeeze()
    a_ind_next = [slice(None)]*dim
    if dims<3:#0,1，见Readme.md解释
        col_ind_next = torch.roll(col_ind_next, shifts, dims)
        a_ind_curr[dims] = 0
        a_ind_next[dims] = N[w]-1
    
    
    a_curr[tuple(a_ind_curr)] = f1
    a_next[tuple(a_ind_next)] = fg

    indices = torch.stack([row_ind.repeat(2), torch.cat([col_ind_curr, col_ind_next.view(-1)])])
    v = torch.cat([-a_curr.view(-1), a_next.view(-1)])
    Dw = torch.sparse_coo_tensor(indices,v,size = (M,M))#COO, CSR, CSC, BSR, and BSC.
    """
    Dw = sparse([row_ind(:); row_ind(:)], [col_ind_curr(:); col_ind_next(:)], ...
            [-a_curr(:); a_next(:)], M, M);
    """
    # a_curr[w,:,:] = f1
    """
    matlab代码
    col_ind_next = reshape(1:M, N);
    与line78对应
    """
    return Dw

def create_spdiag(diag_elements:torch.Tensor):
    # This function creates a sparse diagonal matrix from the provided diagonal elements.
    # 将输入向量转换为一维张量
    vec = diag_elements.view(-1)
    n = vec.size(0)
    
    # 创建行索引和列索引
    indices = torch.arange(n, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    
    # 创建稀疏对角矩阵
    D = torch.sparse_coo_tensor(indices, vec, (n, n))
    return D
    # return sparse.diags(diag_elements)

def mult_vec(dl_factor_cell, dl):
    # This function multiplies the elements of dl_factor_cell with dl.
    # The implementation of this function is not provided in the original MATLAB code.
    # You need to implement this function based on your specific requirements.
    n = len(dl_factor_cell)
    Prod_cell = [None] * n
    for w in range(n):
        assert dl_factor_cell[w].shape == dl[w].shape, RuntimeError("shape of dl_factor_cell and dl should be the same")
        Prod_cell[w] = dl_factor_cell[w] * dl[w]#element-wise multiplication
    return Prod_cell

# Example usage
# Assuming GT, Axis, Sign, Grid2d, Grid3d, create_Dw, and create_spdiag are defined elsewhere
# s = Sign.p  # or Sign.n
# ge = GT.prim  # or GT.dual
# dl_factor_cell = [...]  # Define this based on your requirements
# gridnd = Grid3d(...)  # or Grid2d, define this based on your requirements
# Ds_cell = create_Ds(s, ge, dl_factor_cell, gridnd)