import torch
import torch.nn.functional as F

"""
普通矩阵的差分
"""
def create_diff_matrix_2d(N, axis, boundary_condition='periodic'):
    """
    Create a finite difference matrix for a 2D tensor.
    
    Args:
        N (tuple): The size of the matrix (number of grid points) in each dimension.
        axis (int): The axis along which to compute the finite difference (0 for x, 1 for y).
        boundary_condition (str): The boundary condition ('periodic' or 'dirichlet').
    
    Returns:
        torch.Tensor: The finite difference matrix.
    """
    I = torch.eye(N[axis])
    D = torch.roll(I, shifts=-1, dims=0) - I
    
    if boundary_condition == 'periodic':
        D[-1, 0] = 1
    elif boundary_condition == 'dirichlet':
        D[-1, :] = 0
    else:
        raise ValueError("Unsupported boundary condition. Use 'periodic' or 'dirichlet'.")
    
    if axis == 0:
        D = D.unsqueeze(2).repeat(1, 1, N[1])
    elif axis == 1:
        D = D.unsqueeze(0).repeat(N[0], 1, 1)
    
    return D

def apply_diff_matrix_2d(tensor, D, axis):
    """
    Apply the finite difference matrix to a 2D tensor.
    
    Args:
        tensor (torch.Tensor): The input 2D tensor.
        D (torch.Tensor): The finite difference matrix.
        axis (int): The axis along which to apply the finite difference.
    
    Returns:
        torch.Tensor: The result of applying the finite difference matrix.
    """
    if axis == 0:
        result = torch.matmul(D, tensor)
    elif axis == 1:
        result = torch.matmul(tensor, D)
    else:
        raise ValueError("Unsupported axis. Use 0 for x-axis or 1 for y-axis.")
    
    return result

# 示例用法
N = (5, 5)  # Number of grid points in each dimension
tensor = torch.tensor([[1, 2, 3, 4, 5],
                       [6, 7, 8, 9, 10],
                       [11, 12, 13, 14, 15],
                       [16, 17, 18, 19, 20],
                       [21, 22, 23, 24, 25]], dtype=torch.float32)

# Create finite difference matrices for x and y axes
D_x = create_diff_matrix_2d(N, axis=0, boundary_condition='periodic')
D_y = create_diff_matrix_2d(N, axis=1, boundary_condition='periodic')

# Apply finite difference matrices to the tensor
diff_x = apply_diff_matrix_2d(tensor, D_x, axis=0)
diff_y = apply_diff_matrix_2d(tensor, D_y, axis=1)

print("Original tensor:")
print(tensor)
print("Difference along x-axis:")
print(diff_x)
print("Difference along y-axis:")
print(diff_y)