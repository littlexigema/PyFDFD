import torch.nn as nn
import torch

class CSI_Loss(nn.Module):
    def __init__(self, weight_data:float, weight_state:float):
        super().__init__()
        self.part_data = weight_data
        self.part_state = weight_state
    
    def forward(self, model:nn.Module):
        assert model.E_R.shape == model.E_R_guess.shape,ValueError("the shape of inputs shoud be same!")

        data_error = torch.sum((model.E_R - model.E_R_guess)**2)
        state_error = 
