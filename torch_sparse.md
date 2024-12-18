# torch.sparse

torch.Tensor默认是在物理存储空间连续存储的，但一些数据大多数为0,因此官方提供了性能优化的稀疏存储格式

稀疏存储格式可以看作一种性能优化，但是相对于其他格式，执行时间会有所增加
**简单使用：**
```python
>>> a = torch.tensor([[0, 2.], [3, 0]])
>>> a.to_sparse()
tensor(indices=tensor([[0, 1],
                       [1, 0]]),
       values=tensor([2., 3.]),
       size=(2, 2), nnz=2, layout=torch.sparse_coo)
```

同样也支持其他格式的稀疏矩阵：
PyTorch currently supports COO, CSR, CSC, BSR, and BSC.


详情见：https://pytorch.org/docs/stable/sparse.html