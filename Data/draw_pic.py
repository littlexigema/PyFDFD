import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python3 visualize_npy.py <npy_file>")
    sys.exit(1)
npy_file = sys.argv[1]

#pwd = os.getcwd()

#num = os.listfile(npy_file)
file_names =[item for item in os.listdir(npy_file) if os.path.isdir(os.path.join(npy_file, item)) ]
num = len(file_names)

# 创建图形
fig, axes = plt.subplots(1, num, figsize=(15, 3))
fig.subplots_adjust(right=0.9)

# 读取所有数据并找到全局最大最小值
all_data = []
for i in range(num):
    path = os.path.join(npy_file, file_names[i], 'gt.npy')
    data = np.load(path)
    all_data.append(data)

vmin = min(np.nanmin(d) for d in all_data)
vmax = max(np.nanmax(d) for d in all_data)

# 绘制热图
for i, data in enumerate(all_data):
    im = axes[i].imshow(data, cmap='hot', vmin=vmin, vmax=vmax)
    axes[i].set_title(f'Case {i+1}',weight = 'bold')
    axes[i].axis('off')

# 添加颜色条
cax = plt.axes([0.92, 0.15, 0.02, 0.7])
plt.colorbar(im, cax=cax)

# 保存图像
plt.savefig('output.png', dpi=300, bbox_inches='tight')
plt.close()
