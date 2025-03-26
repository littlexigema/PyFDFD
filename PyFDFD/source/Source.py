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
    
    def __init__(self, lgrid_cell: List[torch.Tensor], 
                 laltgrid_cell: List[torch.Tensor], 
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
        if not (isinstance(lgrid_cell, list) and len(lgrid_cell) == Axis.count()):
            raise ValueError(f'"lgrid_cell" should be length-{Axis.count()} list of tensors')
        
        if not (isinstance(laltgrid_cell, list) and len(laltgrid_cell) == Axis.count()):
            raise ValueError(f'"laltgrid_cell" should be length-{Axis.count()} list of tensors')
        
        # Initialize properties
        self._lgrid = lgrid_cell
        self._laltgrid = laltgrid_cell
        self._shape = shape
        self._gt = None  # Will be set later by set_gridtype
        self._forceprim = forceprim

    @property
    def lgrid(self) -> List[torch.Tensor]:
        """Get grid locations."""
        return self._lgrid

    @property
    def laltgrid(self) -> List[torch.Tensor]:
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

    def get_l(self) -> List[List[torch.Tensor]]:
        """
        Get grid locations based on grid type.

        Returns:
            List of lists containing grid locations for each axis and grid type
        """
        l = [[None for _ in range(GT.count())] for _ in range(Axis.count())]
        
        if self.forceprim:
            for i in range(Axis.count()):
                l[i][GT.PRIM.value] = self.lgrid[i]
                l[i][GT.DUAL.value] = self.laltgrid[i]
        else:
            for i in range(Axis.count()):
                l[i][self.gt.value] = self.lgrid[i]
                l[i][GT.alter(self.gt).value] = self.laltgrid[i]
        
        return l

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

        try:
            return self.generate_kernel(w_axis, grid3d)
        except Exception as e:
            raise RuntimeError('Source assignment failed.') from e

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