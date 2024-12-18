xs = 70
ys = 100
n_T = 18
n_R = 360
m_unit = 1.5e-3

name = "test_circle"#反演物体名称，如果存在相关数据集则加载，否则合成相关chi
regSize=[]#整个反演物理区域大小(并没完全搞懂，这个有什么用)
R_transmitter = 0.72#单位m
R_receiver = 0.76#单位m
C_0 = 299792458
C_norm = 1#离散化后的\Delta x,\Delta t时的C_norm = 1
