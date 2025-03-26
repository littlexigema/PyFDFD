import torch

# 定义一些数学函数、转换等

def hankel_0_1(z):
    """
    相当于MATLAB中的 besselh(0, 1, z)
    J 是第一类Bessel函数, Y 是第二类Bessel函数
    Hankel^(0)_1 (z) = J_0(z) + i * Y_0(z)
    """
    J_nu = torch.special.bessel_j0(z)
    Y_nu = torch.special.bessel_y0(z)
    return J_nu + 1j * Y_nu

def hankel_1_1(z):
    """
    相当于MATLAB中的 besselh(1, 1, z)
    Hankel^(1)_1 (z) = J_1(z) + i * Y_1(z)
    """
    J_nu = torch.special.bessel_j1(z)
    Y_nu = torch.special.bessel_y1(z)
    return J_nu + 1j * Y_nu

def Bessel_J_0(z):
    """
    相当于MATLAB中的 besselj(0, z)
    """
    return torch.special.bessel_j0(z)

def Bessel_J_1(z):
    """
    相当于MATLAB中的 besselj(1, z)
    """
    return torch.special.bessel_j1(z)

def CD_to_ROI(x, ROImask):
    """
    x 要求是已经被flatten的CD
    输出也是flattend的ROI
    """
    return x[ROImask.flatten()]

def ROI_to_CD(x, ROImask):
    """
    x 要求是已经被flatten的ROI
    输出也是flattend的CD
    """
    x_ROI = torch.zeros(ROImask.shape, dtype=x.dtype, device=x.device)
    x_ROI[ROImask] = x
    return x_ROI.flatten()

def comp_mul(x, y):
    """
    用实数表示复数矩阵乘法, 输入是两个实数张量, 输出是实数张量
    第一个维度索引0是实部, 索引1是虚部, 第二三维度是矩阵乘积的维度
    """
    x_Re = x[0]
    x_Im = x[1]
    y_Re = y[0]
    y_Im = y[1]
    return torch.stack([x_Re@y_Re - x_Im@y_Im, x_Re@y_Im + x_Im@y_Re], dim=0)

if __name__ == '__main__':
    # A_comp = torch.tensor([[1.0+2.0j, 3.0+4.0j],
    #               [5.0+6.0j, 7.0+8.0j]])
    # B_comp = torch.tensor([[9.0+10.0j, 11.0+12.0j],
    #               [13.0+14.0j, 15.0+16.0j]])
    # A_real = torch.stack([A_comp.real, A_comp.imag], dim=0)
    # B_real = torch.stack([B_comp.real, B_comp.imag], dim=0)
    # C = comp_mul(A_real, B_real)
    # C1 = A_comp @ B_comp
    # print(C, C1)
    # A_eye = torch.stack([torch.eye(2), torch.eye(2)], dim=0)
    # print(A_eye)
    value = torch.tensor([0.0])
    print(hankel_0_1(value))