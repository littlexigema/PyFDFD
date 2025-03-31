# FDFD微波成像系统开发计划

## 一、验证现有系统

### 1. 电场计算验证
- [ ] 测试合成的入射场($E_i$)与真实入射场($GT_{ei}$)之间的误差
- [ ] 测试FDFD计算的散射场与真实散射场之间的误差

### 2. FDFD正问题的J源添加
- [ ] 完成PointSrc类的初始化
- [ ] 核对PointSrc的generate_kernel函数
- [ ] 优化FWD实现：使用单个FWD配合动态网格，通过不同dl减少计算复杂度

### 3. 入射场计算流程
1. 从build_system获取J源（通过assign_source）
2. 求解方程：
```python
[E, H] = solve_eq_direct(solveropts.eqtype, solveropts.pml, 
                        osc.in_omega0(), eps, mu, s_factor, J, M, grid3d)
```
3. 实现从MatrixEquation到mycreate_eqTM的转换

## 二、动态网格优化

### 1. 网格生成改进
- [ ] 实现直接从gt图像动态生成grid
- [ ] 创建支持更高dl的图片加载shape
- [ ] 优化动态网格生成代码

### 2. 关键特性
- 添加接收源位置到grid
- 引入动态网格直接计算接收处测量场
- 避免使用格林函数

### 3. 代码最终检测
```matlab
% An example of |lprim_part_cell| is
%
%  {[0 0.5], 1, [10 11 12], 1.5, [20.5, 21], 2, [30, 31]}
%
% where |[0.0 0.5]|, |[10 11 12]|, |[20.5, 21]|, |[30, 31]| are subgrids; |1| is
% the target |dl| between |0.5| and |10|; |1.5| is the target |dl| between |12|
% and |20.5|; |2| is the target |dl| between |21| and |30|.  Then,
% |complete_lprim1d(lprim_part_cell)| generates a grid between |0| and |31|.  An
% error is generated when the function fails to generate a primary grid.

%%% Example
%   % Complete a primary grid.
lprim_part_cell = {[0 0.5], 1, [10 11 12], 1.5, [20.5, 21], 2, [30, 31]};
lprim_array = complete_lprim1d(lprim_part_cell);
```
**python生成lprim_part_cell后，**
- [ ] 查看并可视化lprim_part_cell来检查计算复杂度是否降低
- [ ] 动态生成grid是否正确

## 三、未来展望

### 1. 潜在优化方向
- Maxwell-fdfd原生支持动态网格计算系统矩阵
- 研究极坐标系网格的可能性
- 探索网格优化对后续优化的影响

### 2. 近期学习计划
- [ ] 完成Stanford S183课程
- [ ] 完成Stanford S184课程
- [ ] 完成开源代码编写

---

## 补充说明

1. **代码验证重要性**：
   - 神经网络在ECCV上的成功反演不能完全证明E_i, E_s矩阵的正确性
   - 可能受益于网络的鲁棒性

2. **动态网格的优势**：
   - 提高计算效率
   - 减少内存占用
   - 提升精度

3. **技术细节**：
   - complete_lprim1d函数用于生成完整的一维主网格
   - 需要注意网格点的连续性和平滑过渡
   - 保持适当的网格比例关系