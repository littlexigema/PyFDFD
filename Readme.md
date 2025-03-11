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
$\nabla \times \bf{E} = -j \omega \mu_0 H$
$\nabla \times \bf{H} = J + j \omega \epsilon_0 E$


### FDTD Formulation
$\nabla \times \bf{E} = -\frac{\partial B}{\partial t}$
$\nabla \times \bf{B} = \mu_0 J + \mu_0 \epsilon_0 \frac{\partial E}{\partial t}$