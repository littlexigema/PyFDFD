from utils import round_complex_tensor
import torch
import os
pwd = os.getcwd()

xs = 1000#(mm)1m
# xs = 3500#暂时性的修改
#ys = 100#(mm)
ys = xs
assert xs == ys,ValueError("xs should be equal to ys! MOM needs circle")

# m_unit = 1.5e-3
# m_unit = 1/64
# m_unit_for = 3/(96+14)#合成数据用的分辨率，避免inverse crime
# m_unit = 3.5/(96*3)#暂时性的修改

m_unit = 1/32
m_unit_for = m_unit

centre = [0,0,0]

name = "multi_circle"#反演物体名称，如果存在相关数据集则加载，否则合成相关chi
file_path = os.path.join(pwd,'Data','multi_circles','1_ground_truth.npy')
regSize=[]#整个反演物理区域大小(并没完全搞懂，这个有什么用)
TME_mode = "TM"
R_transmitter = 1.5#0.72#单位m
R_receiver = 1.5#0.76#单位m

fre = 0.4#2GHZ
d_pml = 5#像素数
eps_b = 1;sigma_b = 0
eps0 = 8.85418782e-12
C_0 = 299792458
pi = 3.141592653589793
eta_0 = 120*pi
C_norm = 1#离散化后的\Delta x,\Delta t时的C_norm = 1
lambda_ = C_0/(fre*1e9)
k_0 = 2 *pi / lambda_
omega = k_0 * C_0#这个omega仅参与电场的计算，不参与系统矩阵的计算
bool_plane = 0#line source incidence;1:plane wave incidence
Coef = 1j* k_0*eta_0
"""
transmitter和receiver相关参数
"""
n_T = 16
n_R = 32
theta_T = torch.linspace(0,2*pi,n_T+1)[:-1]#transmitter的角度tensor
theta_R = torch.linspace(0,2*pi,n_R+1)[:-1]#receiver的角度tensor
pos_T = torch.polar(torch.tensor([R_transmitter]),theta_T)#单位m
pos_T = round_complex_tensor(pos_T,decimals=2)
pos_T_for = pos_T/m_unit_for
pos_T_back = pos_T/m_unit
xs              = round(xs * 1e-3 / m_unit) * m_unit
ys              = round(ys * 1e-3 / m_unit) * m_unit
R_max           = max(R_transmitter,R_receiver)+0.5
x_for           = round(R_max/m_unit_for) * m_unit_for
invdom = [centre[0]-xs,centre[0]+xs,centre[1]-ys,centre[1]+ys,centre[2],centre[2]]#反演计算域
fordom = [centre[0]-x_for,centre[0]+x_for,centre[1]-x_for,centre[1]+x_for]#合成数据计算域
