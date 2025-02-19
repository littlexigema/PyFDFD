**create_Dw**
函数中，dims这样得到，shifts, dims= -1, w.value^1
w.value = 0,1,2
分别得到的dims = 1,0,3我们只允许前两种调用
col_ind_next = torch.roll(col_ind_next, shifts, dims)
在matlab允许对2Dmatrix更高维度进行circshift,矩阵不发生修改
pytorch这么使用会报错，因此特判