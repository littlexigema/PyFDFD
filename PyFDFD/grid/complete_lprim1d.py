import numpy as np
import torch
from typing import List, Union, Tuple
import math

def find_stiff_ddl(dl_array: torch.Tensor, rt: float) -> torch.Tensor:
    """Find indices where grid spacing changes too abruptly."""
    if len(dl_array) < 2:
        raise ValueError('"dl_array" should be length-2 or longer')
        
    dl1 = dl_array[:-1]
    dl2 = dl_array[1:]
    ratios = dl1/dl2
    
    if rt >= 1:
        ind = torch.nonzero(torch.logical_or(ratios > rt, ratios < 1/rt)).flatten()
    else:
        ind = torch.nonzero(torch.logical_or(ratios > 1/rt, ratios < rt)).flatten()
        
    return ind

issorted = lambda x:not (torch.tensor(x[1:]) < torch.tensor(x[:-1])).any().item()#升序排列
is_smooth = lambda dl_array,rt:len(find_stiff_ddl(dl_array,rt))==0#=0case为smooth

def complete_lprim1d(lprim_part_cell: List[Union[torch.Tensor, float]]) -> torch.Tensor:
    """
    Complete a 1D primary grid by filling gaps between parts of the grid.
    
    Args:
        lprim_part_cell: List where each element is either:
            - a float (grid cell size)
            - a tensor [l_i, ..., l_f] (subgrid between l_i and l_f)
            
    Returns:
        torch.Tensor: Complete 1D primary grid
    """
    # Input validation
    numelems = len(lprim_part_cell)
    if numelems % 2 != 1:
        raise ValueError('"lprim_part_cell" should have odd number of elements')
        
    # Initialize data structures
    ds = []  # List of {subgrid, dl_target} pairs
    
    # Process input cell array
    for i in range((numelems+1)// 2):
        if i == (numelems + 1) // 2 - 1:
            curr = [lprim_part_cell[2*i]]
        else:
            curr = [lprim_part_cell[2*i], lprim_part_cell[2*i + 1]]
            
        # Validate subgrid
        if len(curr[0]) < 2 or (not issorted(curr[0])):
            raise ValueError(f'Element #{2*i+1} should be length-2 or longer tensor with ascending elements')
            
        # Check ordering between subgrids
        if i >= 1:
            assert prev[0][-1] < curr[0][0], ValueError(f'Subgrids {prev[0]} and {curr[0]} should be sorted in ascending order')
                
        ds.append(curr)
        prev = curr

    # Single subgrid case
    if numelems == 1:
        return torch.tensor(ds[0][0])

    # Fill gaps between neighboring subgrids
    rt = 1.9#3.5#1.9  # target ratio of geometric sequence
    rmax = 2.0#3.2#2.0  # maximum ratio of geometric sequence
    
    curr = ds[0]
    lprim_array = curr[0]
    numgrids = len(ds)
    assert numgrids>=2,ValueError('numgrid shoud larger than 1')

    
    for i in range(1, numgrids):
        dl_n = lprim_array[-1] - lprim_array[-2]  # current subgrid dl
        next_grid = ds[i]
        dl_p = next_grid[0][1] - next_grid[0][0]  # next subgrid dl
        gap = torch.tensor([lprim_array[-1], next_grid[0][0]])  # gap between subgrids
        dl_target = curr[1]
        
        try:
            filler = fill_targeted_geometric(dl_n, gap, dl_target, dl_p, rt, rmax)
        except Exception as err:
            raise RuntimeError(f'Grid generation failed between subgrids {curr[0]} and {next_grid[0]} '
                             f'with target dl = {curr[1]}: {str(err)}')
        
        # Preserve provided subgrids despite roundoff errors in filler
        lprim_array = torch.cat([
            torch.tensor(lprim_array),
            filler[1:-1],
            torch.tensor(next_grid[0])
        ])
        curr = next_grid
        
    # Check final grid quality
    dlprim_array = lprim_array[1:] - lprim_array[:-1]
    ind = find_stiff_ddl(dlprim_array, rmax)
    if len(ind) > 0:
        i = ind[0]
        # raise RuntimeError(f'Grid generation failed: grid vertex locations '
        #                  f'[..., {lprim_array[i]}, {lprim_array[i+1]}, {lprim_array[i+2]}, ...] '
        #                  f'are separated by [..., {dlprim_array[i]}, {dlprim_array[i+1]}, ...] '
        #                  f'that do not vary smoothly.')
    
    return lprim_array

def fill_constant(dl_min: float, dl_max: float, gap: torch.Tensor, 
                 rt: float, rmax: float) -> torch.Tensor:
    """
    Fill gap with constant grid spacing.
    
    Args:
        dl_min: Minimum grid spacing
        dl_max: Maximum grid spacing
        gap: Tensor [start, end] defining the gap to fill
        rt: Target ratio for smooth transitions
        rmax: Maximum allowed ratio between adjacent spacings
        
    Returns:
        torch.Tensor: Grid points filling the gap
    """
    # Check dl_min <= dl_max
    if dl_min > dl_max and not torch.isclose(torch.tensor(dl_min), torch.tensor(dl_max)):
        raise ValueError(f'dl_min = {dl_min} should not be greater than dl_max = {dl_max}')

    # Check smoothness between dl_min and dl_max
    if not is_smooth(torch.tensor([dl_min, dl_max]), rt):
        raise ValueError('dl_min and dl_max should be similar')

    # Check gap validity
    L = gap[1] - gap[0]
    if L <= 0:
        raise ValueError('second element of gap should be greater than the first element')

    # Calculate candidate number of cells
    numcells = torch.tensor([torch.floor(L/dl_max), torch.ceil(L/dl_max)])
    dl = L / numcells

    # Check if either grid spacing is valid
    if not (is_smooth(torch.tensor([dl_min, dl[0]]), rmax) or 
            is_smooth(torch.tensor([dl_min, dl[1]]), rmax)):
        raise ValueError(f'dl = {dl_min:.2e} is too small or dl = {dl_max:.2e} '
                       f'is too large for gap size = {L:.2e}')

    # Choose grid spacing - prefer coarser grid unless it fails smoothness check
    if not is_smooth(torch.tensor([dl_min, dl[0]]), rmax):
        numcells = numcells[1]  # finer grid
    else:
        numcells = numcells[0]  # coarser grid

    # Generate evenly spaced points
    return torch.linspace(gap[0], gap[1], int(numcells) + 1)

def fill_targeted_geometric_sym(dl_sym,gap,dl_t,rt,rmax):
    """
    在gap之间生成平滑dl组成的grid，dl_sym代表gap两边的dl一致
    """
    L = gap[1] - gap[0]
    if is_smooth(torch.tensor([dl_sym,dl_t]),rt):
        filler = fill_constant(dl_sym,dl_t,gap,rt,rmax)
        return filler
    else:
        dl_max = dl_t
        dl_min = dl_sym
        n = math.ceil(math.log(dl_max / dl_min) / math.log(rt))  # Smallest n satisfying (dl_max/dl_min)^(1/n) <= rt
        r = (dl_max / dl_min) ** (1 / n)  # Ratio of geometric sequence
        dl_array = dl_min * torch.tensor([r**i for i in range(1, n + 1)])
        
        L_graded = dl_array.sum().item()  # Sum of graded dl's
        """以下代码待检查"""
        if 2 * L_graded > L:
            # Try to geometrically increase dl close to dl_t
            n1 = math.ceil(math.log(L / (2 * dl_min * rt)) / math.log(rt))
            n2 = math.ceil(math.log(L / (2 * dl_min)) / math.log(rt))
            n = min(n1, n2)
            r = (L / (2 * dl_min * n)) ** (1 / n)
            dl_array = dl_min * torch.tensor([r**i for i in range(1, n + 1)])
            dl_filler = torch.cat([dl_array, dl_array.flip(0)])
        else:
            # Slightly under-fill the gap with dl_max and graded dl's
            n_dl_max = math.floor((L - 2 * L_graded) / dl_max)
            dl_max_array = dl_max * torch.ones(n_dl_max)
            L_dl_max = dl_max_array.sum().item()

            # Slightly over-fill the gap with dl_min, graded dl's, and L_dl_max
            n_dl_min = math.ceil((L - 2 * L_graded - L_dl_max) / (2 * dl_min))
            dl_min_array = dl_min * torch.ones(n_dl_min)
            L_dl_min = dl_min_array.sum().item()

            # Update the graded dl's
            L_graded = (L - L_dl_max - 2 * L_dl_min) / 2
            r = (L_graded / (dl_min * n)) ** (1 / n)
            dl_array = dl_min * torch.tensor([r**i for i in range(1, n + 1)])
            dl_filler = torch.cat([dl_min_array, dl_array, dl_max_array, dl_array.flip(0), dl_min_array])

    # Construct filler
    filler = torch.cumsum(torch.cat([torch.tensor([gap[0]]), dl_filler]), dim=0)
    return filler

def fill_targeted_geometric(dl_n,gap,dl_t,dl_p,rt,rmax):
    L = gap[1] - gap[0]
    if dl_n == dl_p:
        filler = fill_targeted_geometric_sym(dl_n,gap,dl_t,rt,rmax)
        return filler
    else:
        # Otherwise, generate graded dl's
        dl_max = max(dl_n, dl_p)
        dl_min = min(dl_n, dl_p)
        n = math.ceil(math.log(dl_max / dl_min) / math.log(rt))  # Smallest n satisfying (dl_max/dl_min)^(1/n) <= rt
        r = (dl_max / dl_min) ** (1 / n)  # Ratio of geometric sequence
        dl_array = dl_min * torch.tensor([r**i for i in range(1, n + 1)])
        
        # Slightly under-fill the gap with dl_max and the above generated graded dl's
        L_graded = dl_array.sum().item()  # Sum of graded dl's
        if L_graded > L:
            raise ValueError(f'dl = {dl_min:.2e} is too small or dl = {dl_max:.2e} is too large for gap size = {L:.2e}')
        
        if dl_n < dl_p:
            # Fill from dl_n to dl_p
            filler_n = torch.cumsum(torch.cat([torch.tensor([gap[0]]), dl_array]), dim=0)
            gap_sym = torch.tensor([filler_n[-1].item(), gap[1]])
            filler_sym = fill_targeted_geometric_sym(dl_p, gap_sym, dl_t, rt, rmax)
            filler = torch.cat([filler_n[:-1], filler_sym])
        else:
            # Fill from dl_p to dl_n
            filler_p = torch.cumsum(torch.cat([torch.tensor([gap[1]]), -dl_array]), dim=0).flip(0)
            gap_sym = torch.tensor([gap[0], filler_p[0].item()])
            filler_sym = fill_targeted_geometric_sym(dl_n, gap_sym, dl_t, rt, rmax)
            filler = torch.cat([filler_sym, filler_p[1:]])
        
        return filler
        

