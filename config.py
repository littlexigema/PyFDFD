xs = 70
ys = 100
n_T = 18
n_R = 360
m_unit = 1.5e-3
centre = [0,0,0]
invdom = [centre[0]-xs,centre[0]+xs,centre[1]-ys,centre[1]+ys,centre[2],centre[2]]

name = "circle"#反演物体名称，如果存在相关数据集则加载，否则合成相关chi
regSize=[]#整个反演物理区域大小(并没完全搞懂，这个有什么用)
R_transmitter = 0.72#单位m
R_receiver = 0.76#单位m

fre = 2#2GHZ
d_pml = 5#像素数
eps_b = 1;sigma_b = 0
eps0 = 8.85418782e-12
C_0 = 299792458
pi = 3.141592653589793
C_norm = 1#离散化后的\Delta x,\Delta t时的C_norm = 1
lambda_ = C_0/(fre*1e9)
omega = 2*pi*C_norm/lambda_