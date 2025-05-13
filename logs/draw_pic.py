
import numpy as np
import matplotlib.pyplot as plt
import sys
if len(sys.argv) < 2:
    print("Usage: python3 visualize_npy.py <npy_file>")
    sys.exit(1)
npy_file = sys.argv[1]

pic = np.load(npy_file)
plt.imshow(pic,cmap = 'hot')
plt.axis('off')
# 保存图像
plt.savefig('output.png', dpi=300, bbox_inches='tight',pad_inches = 0)
plt.close()
