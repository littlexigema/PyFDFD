import senior_code.PE as PE
import senior_code.MLP as MLP
from config import xs,nx,ny
import torch

def generate_coordinates(n, dtype=torch.float32):
    coordinates = torch.zeros((2, n, n), dtype=dtype)
    step = 2 / n

    for i in range(n):
        for j in range(n):
            x = -1 + (j + 1/2) * step
            y = 1 - (i + 1/2 ) * step
            coordinates[0, i, j] = x
            coordinates[1, i,j] = y

    x = coordinates[0, :, :].flatten()
    y = coordinates[1, :, :].flatten()
    coordinates = torch.stack((x, y), dim=1)

    return coordinates

coords_fig = generate_coordinates(64, dtype=torch.float32).reshape(nx, ny, 2)
coords_fig *= xs#单位映射
# rff = PE.RFF(n_dim=64*64, n_features=128, std=1, w=10)
# print(True)
# reg_input = rff(coords_fig).to(args.device)
# model = MLP.MLP(args.layer_sizes, args.activations).to(args.device)
# reg_fig_out = model(reg_input)
# reg_fig_scaled = (reg_fig_out + 1) / 2
# reg_fig = reg_fig_scaled.reshape(N_fig, N_fig, 2)
# TV_loss = (Lossf_TV(reg_fig[:, :, 0]))

