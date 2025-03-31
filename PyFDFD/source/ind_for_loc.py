"""匹配grid上网格"""

import torch
import warnings
from ..base.Axis import Axis
from ..base.GT import GT
from ..grid.Grid3d import Grid3d

def ind_for_loc(loc: float, axis: Axis, gt: GT, grid3d: Grid3d) -> int:
    """
    Find the index of a location in a grid array.
    
    Args:
        loc (float): Location to find in the grid
        axis (Axis): Axis along which to search (X, Y, or Z)
        gt (GT): Grid type (PRIM or DUAL)
        grid3d (Grid3d): 3D grid object containing the grid arrays
        
    Returns:
        int: Index of the location in the grid array
    """
    # Get the grid array for the specified axis and grid type
    grid_array = grid3d.l[axis.value, gt]
    
    # Find exact matches
    matches = torch.nonzero(grid_array == loc)
    
    if len(matches) > 0:
        """
        grid_array是一个向量
        """
        # Exact match found
        return matches[0].item()
    else:
        # Find closest point
        ind = torch.argmin(torch.abs(grid_array - loc)).item()
        
        # Warn about using closest point
        warnings.warn(
            f'{gt.name} grid in {axis.name}-axis of "grid3d" does not have '
            f'location {loc}; closest grid vertex at {grid_array[ind]:.6e} '
            f'will be taken instead.',
            RuntimeWarning
        )
        
        return ind