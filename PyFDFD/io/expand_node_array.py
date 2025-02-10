import torch
from base.Axis import Axis
from base.BC import BC
from grid.Grid3d import Grid3d

def expand_node_array(grid3d: Grid3d, node_array: torch.Tensor) -> torch.Tensor:
    """
    Extend eps and mu to the ghost points considering the boundary conditions.
    """
    for w in Axis.elems():
        ind_gn = [slice(None)] * 3
        ind_gp = [slice(None)] * 3
        if grid3d.bc[w] == BC.p:
            ind_gn[w.value] = grid3d.N[w.value] - 1
            ind_gp[w.value] = 0
        else:
            ind_gn[w.value] = 0
            ind_gp[w.value] = grid3d.N[w.value] - 1
        
        node_array = torch.cat((node_array[tuple(ind_gn)], node_array, node_array[tuple(ind_gp)]), dim=w.value)
    
    return node_array

# Example usage
# Assuming Grid3d, Axis, BC, and other necessary components are defined elsewhere
# grid3d = Grid3d(...)  # Define this based on your requirements
# node_array = torch.randn(grid3d.N.tolist())  # Example node array
# expanded_node_array = expand_node_array(grid3d, node_array)