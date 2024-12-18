import torch
import math
from config import eps0

def build_circle(omega,I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,r,x_c,y_c):
    """
    I,J均是background domain的单边像素个数([-I,I],[-J,J])
    r(m)
    x_c,y_c:pixel
    """
    tau = torch.empty(I*J*4,dtype=torch.complex64)
    epsilon_r = epsilon_rb * torch.ones(2*I,2*J)
    sigma = sigma_b * torch.ones(2*I,2*J)
    x = [i*dx for i in range(-I,I+1)]
    y = [i*dy for i in range(-J,J+1)]
    x_c = x_c*dx
    y_c = y_c*dy
    

    for i in range(-I,I+1):
        for j in range(-J,J+1):
            r_inner = math.sqrt((i-x_c)**2+(j-y_c)**2)
            if r_inner <= r:
                epsilon_r[i,j] = epsilon_robj
                sigma[i,j] = sigma_obj
    epsilon_r.view(-1)
    sigma.view(-1)
    tau[:] = (epsilon_r[:]/epsilon_rb-1)-1j*(sigma[:]-sigma_b)/omega/epsilon_rb/eps0
    return tau