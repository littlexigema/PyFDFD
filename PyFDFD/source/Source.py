from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from ..base.GT import GT
from ..base.Axis import Axis
from ..grid.Grid3d import Grid3d
import torch

class Source(ABC):
    """
    Abstract base class for all electric current sources J.
    """
    
    def __init__(self, lgrid_cell: torch.Tensor, 
                 laltgrid_cell: torch.Tensor, 
                 shape=None, 
                 forceprim: bool = False):
        """
        Initialize Source.

        Args:
            lgrid_cell: List of tensors containing locations of grid planes
            laltgrid_cell: List of tensors containing locations of alternative grid planes
            shape: Shape of source (used to draw source)
            forceprim: Flag to control the behavior of get_l()
        """
        # Validate inputs
        if not (isinstance(lgrid_cell, torch.Tensor) and len(lgrid_cell) == Axis.count()):
            raise ValueError(f'"lgrid_cell" should be length-{Axis.count()} list of tensors')
        
        if not (isinstance(laltgrid_cell, torch.Tensor) and len(laltgrid_cell) == Axis.count()):
            raise ValueError(f'"laltgrid_cell" should be length-{Axis.count()} list of tensors')
        
        # Initialize properties
        self._lgrid = lgrid_cell
        self._laltgrid = laltgrid_cell
        self._shape = shape
        self._gt = None  # Will be set later by set_gridtype
        self._forceprim = forceprim

    @property
    def lgrid(self) -> torch.Tensor:
        """Get grid locations."""
        return self._lgrid

    @property
    def laltgrid(self) -> torch.Tensor:
        """Get alternative grid locations."""
        return self._laltgrid

    @property
    def shape(self):
        """Get source shape."""
        return self._shape

    @property
    def forceprim(self) -> bool:
        """Get forceprim flag."""
        return self._forceprim

    @property
    def gt(self) -> Optional[GT]:
        """Get grid type."""
        return self._gt

    def set_gridtype(self, gt: GT):
        """
        Set the grid type.

        Args:
            gt: Grid type (GT.PRIM or GT.DUAL)
        """
        if not isinstance(gt, GT):
            raise ValueError('"gt" should be instance of GT')
        self._gt = gt
    @property
    def l(self) -> torch.Tensor:
        """
        Get grid locations based on grid type.

        Returns:
            List of lists containing grid locations for each axis and grid type
        """
        """
        function l = get.l(this)
			l = cell(Axis.count, GT.count);
			if this.forceprim
				l(:, GT.prim) = this.lgrid.';
				l(:, GT.dual) = this.laltgrid.';
			else
				l(:, this.gt) = this.lgrid.';
				l(:, alter(this.gt)) = this.laltgrid.';
			end
		end
        """
        l = torch.ones((Axis.count(),GT.count()))*torch.nan
        
        if self.forceprim:
            l[:,GT.PRIM] = self.lgrid
            l[:,GT.DUAL] = self.laltgrid
        else:
            l[:,self.gt] = self.lgrid
            l[:,self.gt.alter()] = self.laltgrid
        
        return l

    def ind_for_loc(self,loc: float, axis: Axis, gt: GT, grid3d: Grid3d) -> int:
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
        grid_array = grid3d.l[axis.value][gt]
        
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
            print(
                f'{gt.name} grid in {axis.name}-axis of "grid3d" does not have '
                f'location {loc}; closest grid vertex at {grid_array[ind]:.6e} '
                f'will be taken instead.',
                RuntimeWarning
            )
            
        return ind

    def generate(self, w_axis: Axis, grid3d: Grid3d) -> Tuple[List, torch.Tensor]:
        """
        Generate source distribution.

        Args:
            w_axis: Working axis
            grid3d: 3D grid object

        Returns:
            Tuple containing index_cell and JMw_patch

        Reference code:

        function [index_cell, JMw_patch] = generate(this, w_axis, grid3d)
            chkarg(istypesizeof(w_axis, 'Axis'), '"w_axis" should be instance of Axis.');
            chkarg(istypesizeof(grid3d, 'Grid3d'), '"grid3d" should be instance of Grid3d.');
                    
            try
                [index_cell, JMw_patch] = this.generate_kernel(w_axis, grid3d);  % Cw_patch: current source
            catch err
                exception = MException('Maxwell:srcAssign', 'Source assignment failed.');
                throw(addCause(exception, err));
            end
        end

        """
        if not isinstance(w_axis, Axis):
            raise ValueError('"w_axis" should be instance of Axis')
        if not isinstance(grid3d, Grid3d):
            raise ValueError('"grid3d" should be instance of Grid3d')
        return self.generate_kernel(w_axis,grid3d)
        # try:
        #     return self.generate_kernel(w_axis, grid3d)
        # except Exception as e:
        #     raise RuntimeError('Source assignment failed.') from e

    @abstractmethod
    def generate_kernel(self, w_axis: Axis, grid3d: Grid3d) -> Tuple[List, torch.Tensor]:
        """
        Abstract method to generate source kernel.

        Args:
            w_axis: Working axis
            grid3d: 3D grid object

        Returns:
            Tuple containing index_cell and JMw_patch
        """
        pass