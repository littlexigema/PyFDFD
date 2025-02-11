import torch
from ..base.Axis import Axis
from ..base.Sign import Sign
from ..base.GT import GT
from ..grid.Grid3d import Grid3d

def generate_s_factor(omega: torch.Tensor | float, grid3d: Grid3d, m=None, R=None):
    """
    Generate s-factor for PML.
    For example, s_factor_cell[Axis.x, GT.dual] is the s-factor multiplied to Ex (or eps_ww).
    """
    # if not isinstance(omega, torch.Tensor) or not torch.is_complex(omega):
    #     raise ValueError('"omega" should be a complex tensor.')
    if not isinstance(grid3d, Grid3d):
        raise ValueError('"grid3d" should be an instance of Grid3d.')

    if m is None:
        m = 4
    if not isinstance(m, (int, float)) or m < 1:
        raise ValueError('element of "m" should be real and at least 1.0.')
    m = torch.tensor(m).expand(Axis.count(), Sign.count())

    if R is None:
        R = torch.exp(torch.tensor(-16.0))  # R = exp(-16) ~= 1e-7
    if not isinstance(R, (int, float)) or not (0 < R <= 1):
        raise ValueError('element of "R" should be real and between 0.0 and 1.0.')
    R = torch.tensor(R).expand(Axis.count(), Sign.count())
    lnR = torch.log(R)  # log() is the natural logarithm

    s_factor_cell = [[None for _ in range(GT.count())] for _ in range(Axis.count())]
    for w in Axis.elems():
        Nw = grid3d.N[w.value]

        lpml = grid3d.lpml[w.value, :]  # locations of PML interfaces
        Lpml = grid3d.Lpml[w.value, :]  # thicknesses of PML

        for g in GT.elems():
            l = grid3d.l[w][g]  # length(l) == Nw, rather than Nw+1.
            ind_pml = [l < lpml[Sign.N], l > lpml[Sign.P]]  # indices of locations inside PML

            s_factor = torch.ones(Nw, dtype=torch.cfloat)
            for s in Sign.elems():
                s_factor[ind_pml[s.value]] = calc_s_factor(omega, torch.abs(lpml[s.value] - l[ind_pml[s.value]]), Lpml[s.value], m[w.value, s.value], lnR[w.value, s.value])
                
            s_factor_cell[w.value][g.value] = s_factor
    
    return s_factor_cell

def calc_s_factor(omega: torch.Tensor, depth: torch.Tensor, Lpml: float, m: float, lnR: float) -> torch.Tensor:
    """
    Calculate the s-factor for a given depth.
    """
    sigma_max = -(m + 1) * lnR / (2 * Lpml)  # -(m+1) ln(R)/(2 eta Lpml), where eta = 1 in the unit of eta_0
    sigma = sigma_max * (depth / Lpml) ** m

    kappa_max = 1
    kappa = 1 + (kappa_max - 1) * (depth / Lpml) ** m

    ma = m
    amax = 0
    a = amax * (1 - depth / Lpml) ** ma

    s_factor = kappa + sigma / (a + 1j * omega)  # s = kappa + sigma/(a + i omega eps), where eps = 1 in the unit of eps_0
    return s_factor

# Example usage
# Assuming Grid3d, Axis, Sign, GT, and other necessary components are defined elsewhere
# grid3d = Grid3d(...)  # Define this based on your requirements
# omega = torch.tensor(1.0 + 1.0j)  # Example omega
# s_factor_cell = generate_s_factor(omega, grid3d)