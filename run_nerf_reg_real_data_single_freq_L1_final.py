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
from loss_fn import L_TV, L_TV_L1, L_TV_frac, L_TV_reg
import matplotlib.pyplot as plt

from run_nerf_helpers_real import *

from config import *
from Forward import Forward_model
FWD = Forward_model()
FWD.get_system_matrix(fre)#主要是计算A_for

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

"""
将FWD系统矩阵转到device上
"""
FWD.A_for = FWD.A_for.to(device)
FWD.A = FWD.A.to(device)
FWD.field.R_mat = FWD.field.R_mat.to(device)
FWD.field.Phi_mat = FWD.field.Phi_mat.to(device)
FWD.field.E_inc = FWD.field.E_inc.to(device)
FWD.field.epsil = FWD.field.epsil.to(device)
DEBUG = False
eta_0 = 120 * np.pi
c = 3e8
eps_0 = 8.85e-12

# bool_plane = 0
# Coef = torch.complex(0.0, k_0 * eta_0)
N_rec =32 #360  # Nb. of Receiver
N_inc = 16#8  # Nb. of Incidence

# i = torch.complex(0.0, 1)


MAX = 1#0.075
Mx = 64
step_size = 2 * MAX / (Mx - 1)
cell_area = step_size ** 2  # the area of the sub-domain


def run_network(inputs, fn, embed_fn):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    outputs_flat = fn(embedded)

    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render(freq, H, W, N_cell, E_inc, Phi_mat, R_mat, input, global_step, network_fn, network_query_fn, perturb=0., raw_noise_std=0.):
    re = {}
    lam_0 = c / (freq * 1e9)
    # lam_0 = 0.75
    k_0 = 2 * np.pi / lam_0
    omega = k_0 * c
    epsilon = network_query_fn(input, network_fn)
    epsilon = epsilon.squeeze(-1)
    if global_step < 1000:
        epsilon = epsilon * (global_step / 1000 * 1.3 + 0.9) + 1
    else:
        epsilon = epsilon * 2.2 + 1
    # epsilon = epsilon_gt
    A = FWD.A#FWD.get_system_matrix_epsil()#*norm#system matrix
    re['epsilon'] = epsilon
    # epsilon_numpy = epsilon.cpu().numpy()
    xi_all =  -1j*omega * (epsilon - 1) * eps_0 * cell_area
    xi_forward = torch.reshape(xi_all.t(), [-1, 1]).to(torch.complex64)
    xi_forward_mat = torch.diag_embed(xi_forward.squeeze(-1))
    J = xi_forward_mat@torch.linalg.inv(torch.eye(N_cell,device=device) - (FWD.field.Phi_mat @ xi_forward_mat))@FWD.field.E_inc
    Es_pred = torch.linalg.solve(A,FWD.osc.in_omega0()**2*J)#计算散射场
    re['R_mat_J'] = R_mat@Es_pred
    # xi_forward = torch.reshape(xi_all.t(), [-1, 1])
    # xi_forward_numpy = xi_forward.cpu().numpy()
    # aa = torch.eye(N_cell)
    # bb = torch.diag_embed(xi_forward.squeeze(-1))
    E_tot = torch.linalg.inv(torch.eye(N_cell) - (Phi_mat @ torch.diag_embed(xi_forward.squeeze(-1)))) @ E_inc
    E_s = R_mat @ torch.diag_embed(xi_forward.squeeze(-1)) @ E_tot
    re['E_s'] = E_s
    return re

def create_isp_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    output_ch = 1
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips, tanh=True).to(device)
    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, network_fn: run_network(inputs, network_fn,
                                                              embed_fn=embed_fn, )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

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
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'network_fn': model,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,default='FDFD_test_final',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=1,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=1000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=20000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--freq", type=float, default=0.4,
                        help='frequency')
    return parser

# parser = config_parser()
# args = parser.parse_args()
# epsilon_gt = np.load(os.path.join(args.datadir, 'epsilon_gt.npy'))
# epsilon_gt = torch.Tensor(epsilon_gt).to(device)
# E_s_sim_real = np.load(os.path.join(args.datadir, 'E_s_sim_real.npy'))
# E_s_sim_imag = np.load(os.path.join(args.datadir, 'E_s_sim_imag.npy'))
# E_s_sim_real = torch.Tensor(E_s_sim_real).to(device)
# E_s_sim_imag = torch.Tensor(E_s_sim_imag).to(device)
def train():
    parser = config_parser()
    args = parser.parse_args()

    # loss_TV = L_TV_reg(TVLoss_weight=1)
    loss_TV = L_TV_L1(TVLoss_weight=1)
    # betamax = 1e2
    # kappa = 1.01
    # smooth_lambda = 0.01
    # beta = 2 * smooth_lambda


    # gt = torch.complex(torch.Tensor(gt_real), torch.Tensor(gt_imag)).to(device)

    # epsilon_gt = np.load(os.path.join(args.datadir, 'epsilon_gt.npy'))
    # epsilon_gt = torch.Tensor(epsilon_gt).to(device)

    # epsilon_bp = np.load(os.path.join(args.datadir, 'result_bp.npy'))
    # epsilon_bp = torch.Tensor(epsilon_bp).to(device)





    # x_dom = np.load(os.path.join(args.datadir, 'x_dom.npy'))
    # y_dom = np.load(os.path.join(args.datadir, 'y_dom.npy'))
    # x_dom = torch.Tensor(x_dom).to(device)
    # y_dom = torch.Tensor(y_dom).to(device)
    # xy_dom = torch.stack([x_dom, y_dom], -1)
    x_dom = FWD.field.x_dom.to(device)
    y_dom = FWD.field.y_dom.to(device)
    xy_dom = torch.stack([x_dom, y_dom], -1)
    #mask = np.load(os.path.join(args.datadir, 'mask.npy'))
    #mask = torch.Tensor(mask).to(device)

    N_cell = Mx * Mx

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
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

    # Move testing data to GPU
    # render_poses = torch.Tensor(render_poses).to(device)
    H = 64
    W = 64
    # coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)  # (H, W, 2)
    # Short circuit if only rendering out from trained model
    if args.render_only:
        testsavedir = os.path.join(basedir, expname,
                                   'renderonly_{}_{:06d}.npy'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            fn_test = render_kwargs_test['network_query_fn']
            output = fn_test(xy_dom, render_kwargs_test['network_fn'])
            if global_step < 1000:
                output = output * (global_step / 1000 * 1.3 + 0.9) + 1
            else:
                output = output * 2.2 + 1
            np.save(testsavedir, output.squeeze(-1).numpy().cpu())
        print('Saved test set')

    # for i in trange(0, 1000):
    #     fn_train = render_kwargs_train['network_query_fn']
    #     output = fn_train(xy_dom, render_kwargs_train['network_fn'])
    #     output = output.squeeze(-1)
    #     optimizer.zero_grad()
    #     # tmp = torch.abs(epsilon_bp-1)+1
    #     # tmp = tmp.cpu().numpy()
    #     loss = img2mse(output, epsilon_gt)
    #     loss.backward()
    #     optimizer.step()
    #     print('pre_loss: ', loss.item())

    testsavedir = os.path.join(basedir, expname, 'testset_000000.npy')
    # os.makedirs(testsavedir, exist_ok=True)
    with torch.no_grad():
        fn_test = render_kwargs_test['network_query_fn']
        output = fn_test(xy_dom, render_kwargs_test['network_fn'])
        if global_step < 1000:
            output = output * (global_step / 1000 * 1.3 + 0.9) + 1
        else:
            output = output * 2.2 + 1
        # epsilon_loss = img2mse(output, epsilon_gt)
        # print('epsilon_loss: ', epsilon_loss.item())
        np.save(testsavedir, output.squeeze(-1).cpu().numpy())
    print('Saved test set')

    N_iters = 4000 + 1
    print('Begin')

    freq = args.freq
    freq_dir = 'freq'
    gt = FWD.field.Es
    gt_real = gt.real.numpy()
    gt_imag = gt.imag.numpy()
    # gt_real = np.load(os.path.join(args.datadir, freq_dir, str(freq), 'E_s_real.npy'))
    # gt_imag = np.load(os.path.join(args.datadir, freq_dir, str(freq), 'E_s_imag.npy'))
    gt_real = torch.Tensor(gt_real).to(device)
    gt_imag = torch.Tensor(gt_imag).to(device)

    # Phi_mat_real = np.load(os.path.join(args.datadir, freq_dir, str(freq), 'Phi_mat_real.npy'))
    # Phi_mat_imag = np.load(os.path.join(args.datadir, freq_dir, str(freq), 'Phi_mat_imag.npy'))
    # Phi_mat = torch.complex(torch.Tensor(Phi_mat_real), torch.Tensor(Phi_mat_imag)).to(device)
    Phi_mat = FWD.field.Phi_mat.to(device)
    # R_mat_real = np.load(os.path.join(args.datadir, freq_dir, str(freq), 'R_mat_real.npy'))
    # R_mat_imag = np.load(os.path.join(args.datadir, freq_dir, str(freq), 'R_mat_imag.npy'))
    # R_mat = torch.complex(torch.Tensor(R_mat_real), torch.Tensor(R_mat_imag)).to(device)
    R_mat = FWD.field.R_mat.to(device)
    # E_inc_real = np.load(os.path.join(args.datadir, freq_dir, str(freq), 'E_inc_real.npy'))
    # E_inc_imag = np.load(os.path.join(args.datadir, freq_dir, str(freq), 'E_inc_imag.npy'))
    # E_inc = torch.complex(torch.Tensor(E_inc_real), torch.Tensor(E_inc_imag)).to(device)
    E_inc = FWD.field.E_inc.to(device)
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()
        # Load data

        #####  Core optimization loop  #####
        re = render(freq, H, W, N_cell=N_cell, E_inc=E_inc, Phi_mat=Phi_mat, R_mat=R_mat, input=xy_dom, global_step=global_step, **render_kwargs_train)

        optimizer.zero_grad()
        # tt = re.real
        E_s = re['E_s'] #* mask
        img_loss = img2mse(E_s.real, gt_real) + img2mse(E_s.imag, gt_imag)
        # sim_exp_loss = img2mse(E_s_sim_real, re['E_s'].real) + img2mse(E_s_sim_imag, re['E_s'].imag)
        # sim_gt_loss = img2mse(E_s_sim_real, gt_real) + img2mse(E_s_sim_imag, gt_imag)
        # E_s_numpy = E_s.cpu().detach().numpy()
        # img_loss = E_s.abs() ** 2
        # img_loss = img2mse(E_s.abs(), gt.abs())

        img_loss = img_loss*(E_s.shape[0] * E_s.shape[1])#/mask.sum()

        # TV_loss = loss_TV(re['epsilon'], beta, h, v)
        # E_s_real = E_s.real.cpu().detach().numpy()
        # gt_imag_numpy = gt_imag.cpu().detach().numpy()
        TV_loss = loss_TV(re['epsilon'])
        # loss = img_loss
        loss = img_loss + 0.005 * TV_loss

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

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}.npy'.format(i))
            testsavedir_img = os.path.join(basedir, expname, 'testset_{:06d}.png'.format(i))
            # os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                fn_test = render_kwargs_test['network_query_fn']
                output = fn_test(xy_dom, render_kwargs_test['network_fn'])
                if global_step < 1000:
                    output = output * (global_step / 1000 * 1.3 + 0.9) + 1
                else:
                    output = output * 2.2 + 1

                # epsilon_loss = img2mse(output, epsilon_gt)
                # print('epsilon_loss: ', epsilon_loss.item())
                output = output.squeeze(-1).cpu().numpy()
                np.save(testsavedir, output)
                sc = plt.imshow(output)
                sc.set_cmap('hot')
                plt.colorbar()
                plt.savefig(testsavedir_img)
                plt.close()
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} freq: {freq} img_loss: {img_loss.item()} TV_loss: {TV_loss.item()}")
        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
