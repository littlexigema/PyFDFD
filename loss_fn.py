# import torch.nn as nn
# import torch

# class CSI_Loss(nn.Module):
#     def __init__(self, weight_data:float, weight_state:float):
#         super().__init__()
#         self.part_data = weight_data
#         self.part_state = weight_state
    
#     def forward(self, model:nn.Module):
#         assert model.E_R.shape == model.E_R_guess.shape,ValueError("the shape of inputs shoud be same!")

#         data_error = torch.sum((model.E_R - model.E_R_guess)**2)
#         state_error = 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return k

			
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d
        
# class L_TV(nn.Module):
#     def __init__(self,TVLoss_weight=1):
#         super(L_TV,self).__init__()
#         self.TVLoss_weight = TVLoss_weight
#
#     def forward(self,x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h =  (x.size()[2]-1) * x.size()[3]
#         count_w = x.size()[2] * (x.size()[3] - 1)
#         h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
#         w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
#         return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # batch_size = x.size()[0]
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h =  (x.size()[0]-1) * x.size()[1]
        count_w = x.size()[0] * (x.size()[1] - 1)
        h_tv = torch.pow((x[1:,:]-x[:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,1:]-x[:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)

class L_TV_L1(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV_L1,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # batch_size = x.size()[0]
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h =  (x.size()[0]-1) * x.size()[1]
        count_w = x.size()[0] * (x.size()[1] - 1)
        h_tv = torch.abs((x[1:,:]-x[:h_x-1,:])).sum()
        w_tv = torch.abs((x[:,1:]-x[:,:w_x-1])).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)

class L_TV_L1_3D(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV_L1_3D,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        # batch_size = x.size()[0]
        h_x = x.size()[0]
        w_x = x.size()[1]
        z_x = x.size()[2]
        count_h = (x.size()[0]-1) * x.size()[1] * x.size()[2]
        count_w = x.size()[0] * (x.size()[1] - 1) * x.size()[2]
        count_z = x.size()[0] * x.size()[1] * (x.size()[2] - 1)

        h_tv = torch.abs((x[1:,:,:]-x[:h_x-1,:,:])).sum()
        w_tv = torch.abs((x[:,1:,:]-x[:,:w_x-1,:])).sum()
        z_tv = torch.abs((x[:,:,1:]-x[:,:,:z_x-1])).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w+z_tv/count_z)

class L_TV_frac(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV_frac, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        # batch_size = x.size()[0]
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h = (x.size()[0] - 1) * x.size()[1]
        count_w = x.size()[0] * (x.size()[1] - 1)
        h_tv = torch.pow(torch.abs(x[1:, :] - x[:h_x - 1, :])+1e-10, 1/2).sum()
        w_tv = torch.pow(torch.abs(x[:, 1:] - x[:, :w_x - 1])+1e-10, 1/2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w)

class L_TV_reg(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV_reg, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x, beta, h, v):
        # batch_size = x.size()[0]
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h = (x.size()[0] - 1) * x.size()[1]
        count_w = x.size()[0] * (x.size()[1] - 1)

        h_tv = torch.pow(torch.cat([x[1:, :], x[0:1, :]], 0)-h, 2).sum()
        w_tv = torch.pow(torch.cat([x[:, 1:], x[:, 0:1]], 1)-v, 2).sum()
        return self.TVLoss_weight * beta * 2 * (h_tv / count_h + w_tv / count_w)

class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k


class EdgeEnhanceSmoothLoss(nn.Module):
    def __init__(self, edge_weight=1.0, smooth_weight=0.1,device='cpu'):
        super(EdgeEnhanceSmoothLoss, self).__init__()
        self.edge_weight = edge_weight
        self.smooth_weight = smooth_weight

        # Sobel 算子核
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
    def forward(self, pred):
        # 确保 Sobel 核在与输入相同的设备上
        # device = pred.device
        
        pred = pred.unsqueeze(0).unsqueeze(0)
        # 计算预测图像的梯度
        grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_y = F.conv2d(pred, self.sobel_y, padding=1)

        # 边缘增强损失：梯度的 L2 范数
        edge_loss = torch.mean(grad_x ** 2 + grad_y ** 2)

        # 平滑损失：总变分正则化
        smooth_loss = torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])) + \
                      torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]))

        # 总损失
        loss = self.edge_weight * edge_loss + self.smooth_weight * smooth_loss
        return loss

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3
