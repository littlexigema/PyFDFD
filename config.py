import os
pwd = os.getcwd()

xs = 1000#(mm)1m
#ys = 100#(mm)
ys = xs
assert xs == ys,ValueError("xs should be equal to ys! MOM needs circle")
n_T = 16
n_R = 32
# m_unit = 1.5e-3
m_unit = 1/32
centre = [0,0,0]

name = "circle"#反演物体名称，如果存在相关数据集则加载，否则合成相关chi
regSize=[]#整个反演物理区域大小(并没完全搞懂，这个有什么用)
TME_mode = "TM"
R_transmitter = 3#0.72#单位m
R_receiver = 3#0.76#单位m

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

xs              = round(xs * 1e-3 / m_unit) * m_unit
ys              = round(ys * 1e-3 / m_unit) * m_unit
invdom = [centre[0]-xs,centre[0]+xs,centre[1]-ys,centre[1]+ys,centre[2],centre[2]]