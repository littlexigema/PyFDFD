from utils import round_complex_tensor
import torch
import os
pwd = os.getcwd()
xs = 150
ys = xs

fre = 4.0
m_unit = 0.15/32
m_unit_for = m_unit

ME_mode = "TM"
R_transmitter = .72#0.72#单位m
R_receiver = .72#0.76#单位m

centre = [0,0,0]

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

n_T = 36
n_R = 72

theta_T = torch.linspace(0,2*pi,n_T+1,dtype = torch.float64)[:-1]#transmitter的角度tensor
theta_R = torch.linspace(0,2*pi,n_R+1,dtype = torch.float64)[:-1]#receiver的角度tensor
pos_T = torch.polar(torch.tensor([R_transmitter]),theta_T.to(torch.float32))#单位m
pos_T = round_complex_tensor(pos_T,decimals=2)
pos_T_for = pos_T/m_unit_for
pos_T_back = pos_T/m_unit
xs              = round(xs * 1e-3 / m_unit) * m_unit
ys              = round(ys * 1e-3 / m_unit) * m_unit
R_max           = max(R_transmitter,R_receiver)+0.5
x_for           = round(R_max/m_unit_for) * m_unit_for
invdom = [centre[0]-xs,centre[0]+xs,centre[1]-ys,centre[1]+ys,centre[2],centre[2]]#反演计算域
fordom = [centre[0]-x_for,centre[0]+x_for,centre[1]-x_for,centre[1]+x_for]#合成数据计算域
nx     = int(xs//m_unit)*2#反演分辨率
ny     = int(ys//m_unit)*2