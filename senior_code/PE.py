import torch
import torch.nn as nn
import math
# Positional Encodings or Position Embeddings
class RFF():
    """
    use Random Fourier Features (RFF) to add positional information to the input feature
    """
    def __init__(self, n_dim, n_features, std=1.0, w=10., dtype=torch.float32):
        """
        n_dim: dimension of the input coordinates
        n_features: number of Fourier features
        std: standard deviation of the Gaussian kernel
        """
        self.n_dim = n_dim
        self.n_features = n_features
        self.std = std
        self.w = w
        if self.n_features % 2 != 0:
            raise ValueError("n_features should be even")
        self.B = torch.randn(self.n_dim, int(self.n_features/2), dtype=dtype) * self.std
    
    def __call__(self, x):
        """
        x: (N, dim) input feature
        c' = [sin(2*pi*x@B), cos(2*pi*x@B)]
        """
        # normalize x to [0, 1]
        x = x / (torch.max(x, dim=0)[0])
        out = self.w * torch.cat([torch.sin(2*torch.pi * x @ self.B), 
                        torch.cos(2*torch.pi * x @ self.B)], dim=-1)
        return out

class LearnableRFF(nn.Module):
    """
    Learnable Random Fourier Features positional encoding.
    """
    def __init__(self, n_dim, n_features, w=10.0, std=1.0):
        """
        Args:
            n_dim: input coordinate dimension (e.g., 2D → 2)
            n_features: total number of output Fourier features (must be even)
            w: frequency scaling factor (like omega_0)
            std: standard deviation of initial random frequencies
        """
        super().__init__()
        assert n_features % 2 == 0, "n_features must be even"

        self.n_dim = n_dim
        self.n_features = n_features
        self.w = w

        # Create a learnable parameter B of shape [n_dim, n_features // 2]
        B = torch.randn(n_dim, n_features // 2) * std
        self.B = nn.Parameter(B)

    def forward(self, x):
        """
        Args:
            x: input coordinates, shape (N, n_dim)
        Returns:
            Fourier encoded features, shape (N, n_features)
        """
        # Optional: normalize x to [0, 1] per-dimension
        x = x / (torch.max(x, dim=0, keepdim=True)[0] + 1e-9)

        # Compute sinusoidal features
        x_proj = 2 * math.pi * x @ self.B  # (N, n_features // 2)
        encoded = self.w * torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (N, n_features)

        return encoded


if __name__ == '__main__':
    # test RFF
    coordinates = torch.rand(100, 2).to(torch.float64)
    rff = RFF(n_dim=coordinates.shape[1], n_features=128, std=1.0)
    nn_inputs = rff(coordinates)
    print(nn_inputs.shape)
