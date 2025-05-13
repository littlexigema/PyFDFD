import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

"""SSIM超参数"""
win_size = 7
data_range = 1.5



# 设置根目录，每个子文件夹为一个类别
root_dir = "./"

# 获取所有类别（子文件夹），并排序
class_dirs = ['GROUND TRUTH','OUR METHOD','INR','BPS','PDA']
#class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

# 按类别读取图像，按文件名排序
images_by_class = []
for class_name in class_dirs:
    class_path = os.path.join(root_dir, class_name)
    image_files = sorted(os.listdir(class_path))
    images = list()
    for f in image_files:
    	img = np.load(os.path.join(class_path, f))
    	if class_name=='PDA':
    		img+=1
    		img = np.rot90(img,k = -1)
    		img = np.flip(img,axis = 1)

    	images.append(img)
    		
#    images = [ for f in image_files]
    images_by_class.append(images)

# 获取维度
num_classes = len(images_by_class)
num_images = len(images_by_class[0])

# 检查所有类别图片数量一致
assert all(len(imgs) == num_images for imgs in images_by_class), "每类图片数量必须一致"

# 创建图像矩阵
fig, axes = plt.subplots(num_images, num_classes, figsize=(3*num_classes, 3*num_images))
plt.subplots_adjust(wspace=0.05, hspace=0.2) 
# 确保axes是2D数组
if num_images == 1:
    axes = axes[np.newaxis, :]

# 计算每行的vmin和vmax
row_vmin = np.zeros(num_images)
row_vmax = np.zeros(num_images)
for row in range(num_images):
    row_images = [np.array(images_by_class[col][row]) for col in range(num_classes)]
    row_vmin[row] = min(img.min() for img in row_images)
    row_vmax[row] = max(img.max() for img in row_images)

# 绘制图像并为每行添加一个colorbar
for row in range(num_images):
    # 创建共享的归一化器
    norm = plt.Normalize(vmin=row_vmin[row], vmax=row_vmax[row])
    
    for col in range(num_classes):
        ax = axes[row, col]
        img_array = np.array(images_by_class[col][row])
        im = ax.imshow(img_array, cmap='hot', norm=norm)
        ax.axis('off')
        if col==0:
        	ax.text(0.5,-0.1,'SSIM/PSNR',transform=ax.transAxes, ha = 'center')
        else:
        	gt = np.array(images_by_class[0][row])
        	ssim_val,_=ssim(gt,img_array,full=True,win_size=win_size,data_range=data_range)
        	psnr_val=psnr(gt, img_array,data_range=data_range)
        	ax.text(0.5, -0.1, r'$\mathbf{%.4f}$ / $\mathbf{%.2f}$'%(ssim_val,psnr_val), transform=ax.transAxes, ha='center')
        if col == 0:
            ax.set_ylabel(f"Image {row+1}", fontsize=10)
        if row == 0:
            ax.set_title(class_dirs[col], fontsize=12)
    
    # 为每行添加colorbar
    divider = make_axes_locatable(axes[row, -1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

plt.tight_layout()
plt.savefig('compare_mnist.png',dpi=500)
plt.show()
