import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from senior_code.PE import RFF,LearnableRFF
from config_fresnel import *
import math

"""
摘自Image Interior,暂时性的使用，后续换成自己设计的网络
"""

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    # embed_kwargs = {
    #             'include_input' : True,
    #             'input_dims' : 2,
    #             'max_freq_log2' : multires-1,
    #             'num_freqs' : multires,
    #             'log_sampling' : True,
    #             'periodic_fns' : [torch.sin, torch.cos],
    # }
    # embedder_obj = Embedder(embed_kwargs)
    embed_kwargs = {
        'n_dim' :2,
        'n_features': 512,#128,
        'std' : 1,
    }
    embedder_obj = RFF(**embed_kwargs)
    # embedder_obj = LearnableRFF(**embed_kwargs)
    return embedder_obj, embedder_obj.n_features
    # embed = lambda x, eo=embedder_obj : eo.embed(x)
    # return embed, embedder_obj.out_dim

class CustomPaddingConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding_value=0):
        super(CustomPaddingConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)  # padding=0，因为我们手动填充
        self.padding_value = padding_value
        self.kernel_size = kernel_size

    def forward(self, x):
        # 计算需要的填充大小
        pad = self.kernel_size // 2
        # 使用指定值进行填充
        x = F.pad(x, (pad, pad, pad, pad), mode='constant', value=self.padding_value)
        # 应用卷积
        x = self.conv(x)
        return x

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, output_ch=1, skips=[4], tanh=None,Max = 2):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.tanh = tanh
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

        self.output_linear = nn.Linear(W, output_ch)
        self.ratio = Max-1
        # self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        # torch.nn.init.constant_(self.conv.weight, 1.0 / 9.0)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, 0], dim=-1)
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = self.output_linear(h)
        if self.tanh is not None:
            # outputs = outputs.reshape(1,1,64,64)
            # outputs = self.conv(outputs).reshape(-1,1)
            # outputs = F.pad(outputs, pad=(1, 1, 1, 1), mode='constant', value=0)
            # outputs = torch.nn.AvgPool2d(kernel_size=3,stride=1)(outputs).reshape(-1,1)
            # outputs = 1+self.ratio*torch.sigmoid(outputs)#0.5 * (torch.tanh(outputs) + 1) + 1 
            # outputs = 1+self.ratio*(torch.tanh(outputs)+1)/2
            # if torch.rand(1).item() < 0.5:
            #     outputs = outputs.reshape(1,1,64,64)
            #     outputs = F.pad(outputs, pad=(1, 1, 1, 1), mode='constant', value=1)
            #     outputs = torch.nn.AvgPool2d(kernel_size=3,stride=1)(outputs).reshape(-1,1)
            # outputs = 0.5 * (torch.tanh(outputs) + 1) + 1  # 1.6 # for circle
            # outputs = 0.5 * torch.sigmoid(outputs) + 1
            outputs = torch.sigmoid(outputs)  # [1, 3] for mnist
        # return F.relu(outputs-1)+1
        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30,Max = 2):
        super(SineLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, output_ch=1, skips=[4], omega_0=30, tanh=None):
        super(SIREN, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.tanh = tanh
        self.omega_0 = omega_0
        self.ratio = Max-1
        self.pts_linears = nn.ModuleList()
        for i in range(D):
            if i == 0:
                layer = SineLayer(input_ch, W, is_first=True, omega_0=omega_0)
            elif i in self.skips:
                layer = SineLayer(W + input_ch, W, omega_0=omega_0)
            else:
                layer = SineLayer(W, W, omega_0=omega_0)
            self.pts_linears.append(layer)

        self.output_linear = nn.Linear(W, output_ch)
        # Final layer: custom init if needed
        with torch.no_grad():
            self.output_linear.weight.uniform_(-math.sqrt(6 / W) / omega_0, math.sqrt(6 / W) / omega_0)

    def forward(self, x):
        input_pts, _ = torch.split(x, [self.input_ch, 0], dim=-1)
        h = x
        for i, layer in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)
            h = layer(h)

        outputs = self.output_linear(h)
        if self.tanh is not None:
            outputs = 1 + self.ratio*torch.sigmoid(outputs)
        return outputs


class VGGReconstruction(nn.Module):
    """
    U-net形状
    """
    def __init__(self):
        super(VGGReconstruction, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # (128, 64, 64) -> (64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),   # (64, 64, 64) -> (64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                   # (64, 64, 64) -> (64, 32, 32)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (64, 32, 32) -> (128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # (128, 32, 32) -> (128, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                    # (128, 32, 32) -> (128, 16, 16)
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),    # (128, 16, 16) -> (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),   # (64, 32, 32) -> (64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),     # (64, 32, 32) -> (32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),    # (32, 64, 64) -> (1, 64, 64)
        )
        
    def forward(self, x):
        # 编码
        x = self.encoder(x)
        # 解码
        x = self.decoder(x)
        return x 

class Vgg(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Vgg, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            CustomPaddingConv2d(in_channel,in_channel,1,1),
            nn.ReLU(),
            CustomPaddingConv2d(in_channel,64,3,1,1),  # (128, 64, 64) -> (64, 64, 64)
            nn.ReLU(),
            CustomPaddingConv2d(64, 64,3,1,1),   # (64, 64, 64) -> (64, 64, 64)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),                   # (64, 64, 64) -> (64, 32, 32)
            CustomPaddingConv2d(64,32,1,1,1),
            nn.ReLU(),
            CustomPaddingConv2d(32,1,1,1,1),
            # nn.ReLU()
            nn.Sigmoid()
        )
        # self.encoder2 = nn.Sequential(
        #     CustomPaddingConv2d(32+in_channel, 32, 1, 1),  # (32+in_channel, 64,64) -> (1, 64, 64)
        #     nn.ReLU(),
        #     CustomPaddingConv2d(32,1,1,1,0),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        x = self.encoder(x)
        return x+1
        # 编码
        # encoder_output = self.encoder(x)
        # skip_connection = torch.cat([x, encoder_output], dim=1)  # 拼接后形状: (32+in_channel, 64, 64)
        
        # # 第二部分编码器
        # output = self.encoder2(skip_connection)  # 输出形状: (1, 64, 64)
        # return output+1



# Model
class NeRF_J(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, output_ch=1, skips=[4], tanh=None):
        """
        """
        super(NeRF_J, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.tanh = tanh
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, 0], dim=-1)
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = self.output_linear(h)
        if self.tanh is not None:
            #outputs = 1.6 * (torch.tanh(outputs) + 1) # for real data
            #outputs = 0.0025 * (torch.tanh(outputs))  # 1.6
            outputs = outputs * 1e-5
        return outputs


class NeRF_3D_real(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, output_ch=1, skips=[4], tanh=None):
        """
        """
        super(NeRF_3D_real, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.tanh = tanh
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, 0], dim=-1)
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = self.output_linear(h)
        if self.tanh is not None:
            outputs = 1.6 * (torch.tanh(outputs) + 1) # for real data
            # outputs = 0.5 * (torch.tanh(outputs) + 1) + 1  # 1.6 # for circle
            # outputs = torch.tanh(outputs)*1.5  # [1, 2] # for circle
            # outputs = torch.sigmoid(outputs)  # [1, 3] for mnist
        return outputs
# Model
class NeRF_E(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, output_ch=1, skips=[4], tanh=None):
        """
        """
        super(NeRF_E, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.tanh = tanh
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, 0], dim=-1)
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = self.output_linear(h)
        if self.tanh is not None:
            outputs = 1.6 * (torch.tanh(outputs) + 1) # for real data
            # outputs = 220 * (torch.tanh(outputs))  # 1.6
        return outputs

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
