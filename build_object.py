import torch
import math
from config import eps0

def build_circle(omega,I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,r,x_c,y_c):
    """
    I,J均是background domain的单边像素个数([-I,I],[-J,J])
    r(m)
    x_c,y_c:pixel
    """
    tau = torch.empty(I*J*4,1,dtype=torch.complex64)
    epsilon_r = epsilon_rb * torch.ones(2*I,2*J)
    sigma = sigma_b * torch.ones(2*I,2*J)
    x = [i*dx for i in range(-I,I)]
    y = [i*dy for i in range(-J,J)]
    x_c = x_c*dx
    y_c = y_c*dy
    

    for i,x_i in enumerate(x):
        for j,y_j in enumerate(y):
            r_inner = math.sqrt((x_i-x_c)**2+(y_j-y_c)**2)
            if r_inner <= r:
                epsilon_r[i,j] = epsilon_robj
                sigma[i,j] = sigma_obj
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    # plt.imshow(epsilon_r.numpy(), cmap='viridis', interpolation='nearest')
    # plt.colorbar(label="Value")  # 添加颜色条
    # plt.title("Heatmap of 2D Tensor")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.show()
    epsilon_r = epsilon_r.view(-1,1)
    sigma = sigma.view(-1,1)
    tau[:,0] = ((epsilon_r/epsilon_rb-1)-1j*(sigma-sigma_b)/omega/epsilon_rb/eps0).squeeze()
    return tau