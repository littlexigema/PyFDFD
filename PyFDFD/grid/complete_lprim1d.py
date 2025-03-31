import numpy as np
import torch
from typing import List, Union, Tuple

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
        if len(curr[0]) < 2 or issorted(curr[0]):
            raise ValueError(f'Element #{2*i+1} should be length-2 or longer tensor with ascending elements')
            
        # Check ordering between subgrids
        if i >= 1:
            assert prev[0][-1] < curr[0][0], ValueError(f'Subgrids {prev[0]} and {curr[0]} should be sorted in ascending order')
                
        ds.append(curr)
        prev = curr

    # Single subgrid case
    if numelems == 1:
        return ds[0][0]

    # Fill gaps between neighboring subgrids
    rt = 1.9  # target ratio of geometric sequence
    rmax = 2.0  # maximum ratio of geometric sequence
    
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
            lprim_array,
            filler[1:-1],
            next_grid[0]
        ])
        curr = next_grid
        
    # Check final grid quality
    dlprim_array = lprim_array[1:] - lprim_array[:-1]
    ind = find_stiff_ddl(dlprim_array, rmax)
    if len(ind) > 0:
        i = ind[0]
        raise RuntimeError(f'Grid generation failed: grid vertex locations '
                         f'[..., {lprim_array[i]}, {lprim_array[i+1]}, {lprim_array[i+2]}, ...] '
                         f'are separated by [..., {dlprim_array[i]}, {dlprim_array[i+1]}, ...] '
                         f'that do not vary smoothly.')
    
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
    L = gap[1] - gap[0]
    if is_smooth(torch.tensor([dl_sym,dl_t]),rt):
        filler = fill_constant()

def fill_targeted_geometric(dl_n,gap,dl_t,dl_p,rt,rmax):
    L = gap[1] - gap[0]
    if dl_n == dl_p:
        fillter = fill_targeted_geometric_sym(dl_n,gap,dl_t,rt,rmax)
    

