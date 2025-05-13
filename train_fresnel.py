import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from use_senior_code import *
from loss_fn import L_TV, L_TV_L1, L_TV_frac, L_TV_reg,EdgeEnhanceSmoothLoss
import matplotlib.pyplot as plt

# from config_fresnel import *
from synthesis_or_measure import *
from Forward import Forward_model
classes = "fresnel_two_diel"#'AU'#'multi_shapes'
# name_ = '3/'#'1/'#'ground_truth.npy'#'1/gt.npy'
FWD = Forward_model(inverse=True)
FWD.get_system_matrix(fre)#主要是计算A_for
# FWD.field.set_chi(load_from_gt=True,file = f'./Data/{classes}/{name_}gt.npy')
# FWD.field.set_scatter_E_Born_Appro(FWD.A)#这两行来自于train.py，用来合成数据

import pandas as pd
pwd = os.getcwd()
path = os.path.join(pwd,'Data/fresnel-1')
data = pd.read_csv(os.path.join(path,'twodielTM_4f.txt'),header = None,sep = '\s+')

E_t = torch.zeros((n_R,n_T),dtype = torch.complex64)
E_i = torch.zeros((n_R,n_T),dtype = torch.complex64)
fre_ = torch.tensor(data.iloc[:,2].tolist())
bool_fre = (fre_==fre)

index_T = torch.tensor(data.iloc[:,0].tolist())[bool_fre]-1
index_R = torch.tensor(data.iloc[:,1].tolist())[bool_fre]-1

E_inc_real = torch.tensor(data.iloc[:,3].tolist())[bool_fre]
E_inc_imag = torch.tensor(data.iloc[:,4].tolist())[bool_fre]
E_t_real = torch.tensor(data.iloc[:,5].tolist())[bool_fre]
E_t_imag = torch.tensor(data.iloc[:,6].tolist())[bool_fre]
E_t[index_R,index_T] = E_t_real+1j*E_t_imag
E_i[index_R,index_T] = E_inc_real + 1j*E_inc_imag

bool_mask = (E_t!=0)

from torch.utils.tensorboard import SummaryWriter

from model_fresnel import *
from Unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

"""
将FWD系统矩阵转到device上
"""
R_mat = FWD.field.get_Rmat(FWD.osc.in_omega0()).to(torch.complex64)

"""校准系数"""
calibration = FWD.field.get_calibration(E_i)
#path_R_mat = f'./Data/{classes}/{name_}'
#R_mat_real = np.load(os.path.join(path_R_mat, 'R_mat_real.npy')).astype(np.float64)
#R_mat_imag = np.load(os.path.join(path_R_mat, 'R_mat_imag.npy')).astype(np.float64)
#R_mat = torch.complex(torch.Tensor(R_mat_real), torch.Tensor(R_mat_imag)).to('cpu')

FWD.A_for = FWD.A_for.to(device)
FWD.A = FWD.A.to(device)
# FWD.field.epsil = FWD.field.epsil.to(device)
FWD.field.E_inc = FWD.field.E_inc.to(device)
coords_fig = coords_fig.to(device)
calibration = calibration.to(device)
bool_mask = bool_mask.to(device)

DEBUG = False
eta_0 = 120 * np.pi
c = 3e8
eps_0 = 8.85e-12

# bool_plane = 0
# Coef = torch.complex(0.0, k_0 * eta_0)
N_rec = 72
N_inc = 36#8#16#8  # Nb. of Incidence

# i = torch.complex(0.0, 1)


# MAX = 1
MAX = 0.15/2
Mx = 64
step_size = 2 * MAX / (Mx - 1)
cell_area = step_size ** 2  # the area of the sub-domain


def run_network(inputs, fn, embed_fn):
    """Prepares inputs and applies network 'fn'.
    """
    #NeRF对应代码
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])#(4096,2)
    embedded = embed_fn(inputs_flat)                           #(4096,128)

    outputs_flat = fn(embedded)                                #(4096,2)

    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs
    #U-net Vgg对应代码
    # inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    # embedded = embed_fn(inputs_flat).reshape(nx,ny,-1).permute(2,0,1).unsqueeze(0)

    # outputs_flat = fn(embedded).view(-1)#(2,4096)

    # outputs = outputs_flat.reshape(nx,ny,1)
    # return outputs


def render(freq, H, W, N_cell, E_inc, Phi_mat, R_mat, input,input_J,  network_fn, network_fn_J, network_query_fn,global_step):
    re = {}
    lam_0 = c / (freq * 1e9)
    # lam_0 = 0.75
    k_0 = 2 * np.pi / lam_0
    omega = k_0 * c
    epsilon = network_query_fn(input, network_fn)
    epsilon = epsilon.squeeze(-1)
    
    re['epsilon'] = epsilon#从line 73可以看出，epsilon输出只有实部，也就是epsilon只有虚部
    FWD.guess.set_epsil(epsilon)
    FWD.field.set_chi(load_from_gt=False,guess = FWD.guess)

    A = FWD.A#FWD.get_system_matrix_epsil()#*norm#system matrix
    FWD.field.set_scatter_E_Born_Appro(A)#计算散射场，使用入射场近似估计总电场
    
    Es_pred = FWD.field.E_scat
    # chi = epsilon.reshape(-1,1)-1
    # Es_pred = torch.linalg.solve(A,FWD.osc.in_omega0()**2*chi*FWD.field.E_inc)#计算散射场
    re['Es_pred'] = R_mat @ Es_pred
    # re['R_mat_J'] = R_mat@Es_pred
    # epsilon_numpy = epsilon.cpu().numpy()
    # xi_all = torch.complex(torch.Tensor([0.0]), -omega * (epsilon - 1) * eps_0 * cell_area)
    # xi_forward = torch.reshape(xi_all.t(), [-1, 1])
    # xi_forward_mat = torch.diag_embed(xi_forward.squeeze(-1))
    # xi_E_inc = xi_forward_mat @ E_inc#xi_forward_mat = Diag(\epsilon)
    # print('xi_for')
    # re['J_state'] = xi_E_inc + xi_forward_mat @ Phi_mat @ J#Phi_mat中应该带的An：cell_area放在了xi_all中

    # re['norm_xi_E_inc'] = torch.mean(xi_E_inc.real ** 2 + xi_E_inc.imag ** 2)
    # xi_forward_numpy = xi_forward.cpu().numpy()
    # aa = torch.eye(N_cell)
    # bb = torch.diag_embed(xi_forward.squeeze(-1))
    # E_tot = torch.linalg.inv(torch.eye(N_cell) - (Phi_mat @ torch.diag_embed(xi_forward.squeeze(-1)))) @ E_inc
    # E_s = R_mat @ torch.diag_embed(xi_forward.squeeze(-1)) @ E_tot
    # re['E_s'] = E_s
    return re

def create_isp_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    output_ch = 1
    skips = []
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=[], tanh=True,Max = args.Max).to(device)
    # model = SIREN(D=args.netdepth, W=args.netwidth,
    #              input_ch=input_ch, output_ch=output_ch, skips=[], tanh=True).to(device)
    
    #model = VGGReconstruction()
    #model = Vgg(in_channel=128,out_channel=1)
    # model = UNet(n_channels=128, n_classes=1, bilinear=False)
    grad_vars = list()
    # grad_vars = list(embed_fn.parameters())
    grad_vars += list(model.parameters())

    model_J = NeRF_J(D=args.netdepth, W=args.netwidth,
                   input_ch=input_ch * 2, output_ch=2, skips=skips, tanh=True).to(device)
    # grad_vars += list(model_J.parameters())


    network_query_fn = lambda inputs, network_fn: run_network(inputs, network_fn,
                                                              embed_fn=embed_fn, )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(params = grad_vars,lr = args.lrate)
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 :#and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        model_J.load_state_dict(ckpt['network_fn_J_state_dict'])
    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'network_fn': model,
        'network_fn_J': model_J,
    }

    # NDC only good for LLFF-style forward facing data

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,default=f"FDFD_{classes}_PE_n_features_512_tanh",
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    # parser.add_argument("--datadir", type=str, default='./data/llff/fern',
    #                     help='input data directory')

    # training options
    parser.add_argument("--Max", type=float, default=2.,
                        help='Max value network can predict')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--lrate", type=float, default=5e-3,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')

    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=1,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10,
                        help='frequency of testset saving')

    # parser.add_argument("--noise_ratio", type=float, default=0.05,
    #                     help='noise_ratio')
    parser.add_argument("--sample_perturb", type=float, default=0.005,
                        help='random sample perturbation')

    return parser


def train():
    global R_mat
    parser = config_parser()
    args = parser.parse_args()

    # loss_TV = L_TV_reg(TVLoss_weight=1)
    loss_TV = L_TV_L1(TVLoss_weight=1)
    loss_TV_frac = L_TV_frac(TVLoss_weight=1)
    EnhanceLoss = EdgeEnhanceSmoothLoss(device = device)


    x_t = FWD.field.pos_T.real.flatten()
    y_t = FWD.field.pos_T.imag.flatten()
    xy_t = torch.stack([x_t,y_t],dim=1).to(device)
    



    N_cell = Mx * Mx

    # Load data
    freq = 4.#5#0.3#0.5#0.4#5#0.4
    assert fre == freq,RuntimeError('the frequency generating data is not equal to the inverse one')
    # gt = np.load(os.path.join(args.datadir, '1_Es.npy'))
    gt = E_t-E_i
    # gt = FWD.field.E_scat
    # gt = R_mat @ gt
    # gt_real = gt.real.numpy()
    # gt_imag = gt.imag.numpy()
    # print("gt_real.shape:{}".format(gt_real.shape))
    # if args.noise_ratio != 0:
    #     energe = np.sqrt(np.mean((gt_real ** 2 + gt_imag ** 2))) * (1 / np.sqrt(2))
    #     # gt_real = gt_real + energe*args.noise_ratio*np.random.randn(N_rec, Ninc)
    #     # gt_imag = gt_imag + energe * args.noise_ratio * np.random.randn(N_rec, N_inc)
    #     gt_real = gt_real + energe*args.noise_ratio*np.random.randn(R_mat.shape[0], N_inc)
    #     gt_imag = gt_imag + energe * args.noise_ratio * np.random.randn(R_mat.shape[0], N_inc)
    gt_real = gt.real.to(device)
    gt_imag = gt.imag.to(device)
    R_mat = R_mat.to(device)

    E_inc = FWD.field.E_inc.to(device)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    writer = SummaryWriter(os.path.join(basedir, expname, 'tb'))
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_isp_nerf(args)
    global_step = start


    H = Mx
    W = Mx

    # if args.render_only:
    #     testsavedir = os.path.join(basedir, expname,
    #                                'renderonly_{}_{:06d}.npy'.format('test' if args.render_test else 'path', start))
    #     os.makedirs(testsavedir, exist_ok=True)
    #     with torch.no_grad():
    #         fn_test = render_kwargs_test['network_query_fn']
    #         output = fn_test(xy_dom, render_kwargs_test['network_fn'])
    #         np.save(testsavedir, output.squeeze(-1).numpy().cpu())
    #     print('Saved test set')


    testsavedir = os.path.join(basedir, expname, 'testset_000000.npy')
    # os.makedirs(testsavedir, exist_ok=True)
    with torch.no_grad():
        fn_test = render_kwargs_test['network_query_fn']
        output = fn_test(coords_fig, render_kwargs_test['network_fn'])
        np.save(testsavedir, output.squeeze(-1).cpu().numpy())
    print('Saved test set')

    N_iters = 40000 + 1
    N_epsil = 0 + 1 
    print('Begin')

    start = start + 1


    for i in trange(start, N_iters):
        # xy_dom_random = xy_dom + torch.randn_like(xy_dom) * args.sample_perturb
        # xy_dom_random = coords_fig
        # xy_dom_random = xy_dom
        # coords_inc = torch.cat(
        #     (torch.reshape(xy_dom_random.transpose(0, 1), [-1, 2]).unsqueeze(-2).repeat([1, N_inc, 1]),
        #      xy_t.unsqueeze(0).repeat([N_cell, 1, 1])), -1)

        #####  Core optimization loop  #####
        # re = render(freq, H, W, N_cell=N_cell, E_inc=E_inc, Phi_mat=Phi_mat, R_mat=R_mat, input=xy_dom, input_J=coords_inc, **render_kwargs_train)
        re = render(freq, H, W, N_cell=N_cell, E_inc=E_inc, Phi_mat=None, R_mat=R_mat, input=coords_fig, input_J=None, **render_kwargs_train,global_step=global_step)

        optimizer.zero_grad()
        # tt = re.real
        # img_loss_data = (img2mse(re['R_mat_J'].real, gt_real) + img2mse(re['R_mat_J'].imag, gt_imag))/torch.mean(gt_real **2 + gt_imag **2)
        Es_pred = re['Es_pred']*calibration*bool_mask
        
        img_loss_data = (img2mse(Es_pred.real, gt_real) + img2mse(Es_pred.imag, gt_imag))/torch.mean(gt_real **2 + gt_imag **2)
        # re['Es_pred']
        # img_loss_state = (img2mse(re['J_state'].real, re['J'].real) + img2mse(re['J_state'].imag, re['J'].imag))/(re['norm_xi_E_inc'])

        # TV_loss = loss_TV(re['epsilon'])
        # TV_loss_frac = loss_TV_frac(re['epsilon'])
        edge_loss = EnhanceLoss(re['epsilon'])
        # loss = img_loss
        if global_step <= 1000:
            loss = img_loss_data#(img_loss_data + img_loss_state)
        else:
            loss = img_loss_data#+ 0.01*edge_loss#(img_loss_data + img_loss_state + 0.01*TV_loss)
        loss.backward()
        optimizer.step()
        # beta = min(beta * kappa, betamax)
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################


        # Rest is logging
        writer.add_scalar("loss_data", img_loss_data, i)
        # writer.add_scalar("loss_state", img_loss_state, i)
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fn_J_state_dict': render_kwargs_train['network_fn_J'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}.npy'.format(i))
            testsavedir_img = os.path.join(basedir, expname, 'testset_{:06d}.png'.format(i))
            # os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                fn_test = render_kwargs_test['network_query_fn']
                output = fn_test(coords_fig, render_kwargs_test['network_fn'])
                # print('epsilon_loss: ', epsilon_loss.item())
                output = output.squeeze(-1).cpu().numpy()
                np.save(testsavedir, output)
                sc = plt.imshow(output)
                sc.set_cmap('hot')
                plt.colorbar()
                plt.savefig(testsavedir_img)
                plt.savefig('current_espil.png')
                plt.close()
            print('Saved test set')

        if i % args.i_print == 0:
            # tqdm.write(f"[TRAIN] Iter: {i} freq: {freq} img_loss_data: {img_loss_data.item()} img_loss_state: {img_loss_state.item()}  TV_loss: {TV_loss.item()}")
            # tqdm.write(f"[TRAIN] Iter: {i} freq: {freq} img_loss_data: {img_loss_data.item()} TV_loss: {TV_loss.item()}")
            tqdm.write(f"[TRAIN] Iter: {i} freq: {freq} img_loss_data: {img_loss_data.item()}") #EdgeEnhanceSmoothLoss: {edge_loss.item()}")
            # tqdm.write(f"[TRAIN] Iter: {i} freq: {freq} img_loss_data: {img_loss_data.item()} TV_loss_frac: {TV_loss_frac.item()}")
        global_step += 1
    # FWD.guess.set_epsil(FWD.field.epsil)
    # FWD.field.set_chi(load_from_gt=False,guess = FWD.guess)
    epsil = FWD.guess.epsil.clone().detach()
    chi = epsil.reshape(-1,1)-1
    chi.requires_grad = True  # Enable gradient computation for epsil
    A = FWD.A
    optimizer = torch.optim.SGD(params=[chi], lr=0.5)  # Use epsil as the parameter for SGD
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # Scheduler added
    for i in trange(N_iters+1,N_iters + N_epsil):
        """
        让epsil直接梯度更新
        """
        optimizer.zero_grad()
        B = FWD.osc.in_omega0()**2*chi*FWD.field.E_inc
        B = B.to(torch.complex64)
        Es_pred = torch.linalg.solve(A,B)#计算散射场
        Es_pred = R_mat @ Es_pred
        img_loss_data = (img2mse(Es_pred.real, gt_real) + img2mse(Es_pred.imag, gt_imag))/torch.mean(gt_real **2 + gt_imag **2)
        loss = img_loss_data
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate
        
        if i % args.i_print == 0:
            # tqdm.write(f"[TRAIN] Iter: {i} freq: {freq} img_loss_data: {img_loss_data.item()} img_loss_state: {img_loss_state.item()}  TV_loss: {TV_loss.item()}")
            # tqdm.write(f"[TRAIN] Iter: {i} freq: {freq} img_loss_data: {img_loss_data.item()} TV_loss: {TV_loss.item()}")
            tqdm.write(f"[TRAIN] Iter: {i} freq: {freq} img_loss_data: {img_loss_data.item()}")
        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}.npy'.format(i))
            testsavedir_img = os.path.join(basedir, expname, 'testset_{:06d}.png'.format(i))
            testsavedir_img_2 = os.path.join(basedir, expname, 'current_epsil.png')
            
            # os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                output = (chi+1).reshape(64,64).cpu().numpy()
                np.save(testsavedir, output)
                sc = plt.imshow(output)
                sc.set_cmap('hot')
                plt.colorbar()
                plt.savefig(testsavedir_img_2)
                plt.savefig(testsavedir_img)
                plt.close()
            print('Saved test set')

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # FWD.get_system_matrix_epsil(fre)
    train()


