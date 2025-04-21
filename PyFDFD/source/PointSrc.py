from typing import List, Tuple, Optional, Union
from ..base.Axis import Axis
from .Source import Source
from ..shape import Point
from ..grid.Grid3d import Grid3d
# from . import ind_for_loc
import torch

class PointSrc(Source):
    """
    Point Source class representing a localized electric dipole.
    
    Args:
        polarization_axis (Axis): Direction of polarization (Axis.X, Axis.Y, or Axis.Z)
        location (List[float]): [x, y, z] coordinates of the point source
        I (complex, optional): Current amplitude. Defaults to 1.0
    """
    
    def __init__(self, 
                 polarization_axis: Axis, 
                 location: List[float], 
                 I: complex = 1.0):
        # Validate inputs
        print('created source at position:{}'.format(location))
        if not isinstance(polarization_axis, Axis):
            raise ValueError('"polarization_axis" should be instance of Axis')
            
        if not (isinstance(location, list) and len(location) == Axis.count()):
            raise ValueError(f'"location" should be length-{Axis.count()} list with real elements')
            
        # if isinstance(location, list):
        #     location = torch.tensor(location, dtype=torch.float32)
            
        if not isinstance(I, complex):
            I = complex(I)
            
        # Initialize grid locations
        lgrid = torch.ones(Axis.count())*torch.nan
        laltgrid = torch.ones(Axis.count())*torch.nan
        
        for w in Axis.elems():
            """
            Yee grid上，
            对于一个点源，其极化方向的电流密度（J）需要位于对偶网格点上，
            而垂直于极化方向的电流密度需要位于主网格点上，以确保数值计算的正确性和一致性。
            参考的maxwellfdfd包错误地将polarization 坐标放在dual上了，
            """
            if w == polarization_axis:
                lgrid[w.value] = location[w.value]
            else:
                laltgrid[w.value] = location[w.value]
                
        # Create point shape (assuming you have a Point class similar to MATLAB)
        point = Point(location)  # You'll need to implement this class
        
        # Initialize parent class
        super().__init__(lgrid, laltgrid, point)
        
        # Store properties
        self._polarization = polarization_axis
        self._location = location
        self._I = I
        
    @property
    def polarization(self) -> Axis:
        return self._polarization
        
    @property
    def location(self) -> torch.Tensor:
        return self._location
        
    @property
    def I(self) -> complex:
        return self._I
        
    def generate_kernel(self, w_axis: Axis, grid3d: Grid3d) -> Tuple[List, Optional[torch.Tensor]]:
        """
        Generate the source kernel for the given axis and grid.
        
        Args:
            w_axis (Axis): Working axis
            grid3d (Grid3d): 3D grid object
            
        Returns:
            index_cell : source在grid的索引
            JMw_patch : source的电流密度
            tuple: (index_cell, JMw_patch)

        Description:
            判断w_axis是否等于self.polarization,也就是只计算self.polarization方向的电流密度
        """
        index_cell = [None] * Axis.count()
        
        # Get cyclic permutation of axes
        q, r, p = self.polarization.cycle()  # q, r: axes normal to polarization axis p
        
        if w_axis == p:
            """
            遍历所有Axis,找到location与grid(prim/dual)最近的loc索引
            """
            for v in Axis.elems():
                l = self.location[v.value]
                g = self.gt.alter() if v == p else self.gt
                iv = self.ind_for_loc(l, v, g, grid3d)
                index_cell[v.value] = iv
                
            dq = grid3d.dl[q.value][self.gt][index_cell[q.value]]#找到对应位置的离散化步长
            dr = grid3d.dl[r.value][self.gt][index_cell[r.value]]
            JMw_patch = self.I / (dq * dr)  # I = J * (area)
        else:  # w_axis == q or r
            JMw_patch = None
            
        return index_cell, JMw_patch