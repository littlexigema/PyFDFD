# 关于k的计算
## 一些物理公式
$\lambda = \frac{C}{f}$
$\omega = 2\pi f =\frac{2\pi}{T} = 2\pi \frac{C}{\lambda}$
$k = \frac{2\pi}{\lambda} = \frac{\omega}{C}$
波数就是空间上的角频率

## CC-CSI代码中离散化的操作：
$wvlenn = \frac{\lambda}{m_{unit}}$
本应该这么做:$omega = \frac{2\pi C}{wvlenn} = \frac{2\pi C}{\lambda}\times m_{unit} = \frac{2\pi C}{C}\times f\times m_{unit}$
``` matlab
function omega = in_omega0(this)
    omega = 2*pi / this.wvlen;  % omega = angular_freq / unit.omega0
end
```
为什么？
## 离散化
离散化是在时间和空间上离散，我们选取合适的$\Delta x$和$\Delta t$使得离散化后的速度$C_{normalized} \times \Delta t = \Delta x$，即$C_{normalized} = 1$
故在代码中是将$C = 1$代入，在这种情况下是没有问题的

## 后续可能的改进
目前使用prim,dual两个网格计算FDFD Amatrix,
后续可以使用[Lecture-4e-FDFD-Implementation.pdf](../Lecture-4e-FDFD-Implementation.pdf)改进网格计算(**maybe**)

## generate_lprim3d值只完成了withuniform=True情况的代码正确性验证
## matlab中构建了两个domain，第一个是regSize区域，第二个是真正的逆问题求解区域，我们暂时只关注后者，因此config.py中没有regSize参数

## MaxWell Function
$\nabla \times \bf{E} = -\frac{\partial B}{\partial t}$
$\nabla \times \bf{B} = \mu_0 J + \mu_0 \epsilon_0 \frac{\partial E}{\partial t}$
$\nabla \cdot E = \frac{\rho_0}{\epsilon_0}$
$\nabla \cdot B = 0$
$\bf{B} = \mu H$
$\mu = \mu_0*\mu_r$
$\epsilon = \epsilon_r*\epsilon_0$
### FDFD Formulation
$\nabla \times \bf{E} = -j \omega \mu \bf{H} - \bf{M}$
$\nabla \times \bf{H} = J + j \omega \epsilon_0 E$


### FDTD Formulation
$\nabla \times \bf{E} = -\frac{\partial B}{\partial t}$
$\nabla \times \bf{B} = \mu_0 J + \mu_0 \epsilon_0 \frac{\partial E}{\partial t}$

### Yee Grid
可以从Yee_grid.png中看出网格是如何分布的，TM mode(Ez)模式下，电场E分布在主网格(prim)上,H/B分布在副网格上(dual)

本项目进行的是2D dielectric 电磁反演，transmitter发送TM模式波，receiver实际上测量的是电场E的z分量。

### E-Field
我已经改成使用FDFD AE = B计算散射场，但仍有个问题：虽然正反问题均用FDFD计算，但合成数据和测量数据这两个领域仍不能同时求解，因为FDFD计算出来的测量场E-field是全部grid上的，但实际测量场E-field是圆形边界上的值。

能否使用极坐标系？或者能否使用格林函数映射过去？