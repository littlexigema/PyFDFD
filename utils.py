import torch

def round_complex_tensor(tensor, decimals=4):
    """
    对 complex tensor 的实部和虚部分别保留指定的小数位数。
    
    Args:
        tensor (torch.Tensor): 复数张量。
        decimals (int): 保留的小数位数。
    
    Returns:
        torch.Tensor: 保留指定小数位数的复数张量。
    """
    factor = 10 ** decimals
    real_part = torch.round(torch.real(tensor) * factor) / factor
    imag_part = torch.round(torch.imag(tensor) * factor) / factor
    return real_part + 1j * imag_part