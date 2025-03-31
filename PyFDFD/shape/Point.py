from typing import List, Callable
import torch
from ..base.Axis import Axis
from . import Shape
# from .ZeroVolShape import ZeroVolShape

class Point(Shape):
    """
    Concrete subclass of ZeroVolShape representing a point.
    
    This class represents the shape of a point. It does not have a volume,
    but it is used to force a user-defined primary grid point.
    
    Args:
        location (List[float]): Location of the point in [x, y, z] format
    """
    
    def __init__(self, location: List[float]):
        # Validate location
        if not (isinstance(location, list) and len(location) == Axis.count()):
            raise ValueError(f'"location" should be length-{Axis.count()} list with real elements')
            
        # if isinstance(location, list):
        #     location = torch.tensor(location, dtype=torch.float32)
            
        # Define level set function
        def lsf(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            """
            Level set function for point.
            
            Args:
                x, y, z: Coordinate arrays
                
            Returns:
                torch.Tensor: Level set values
            """
            # Input validation
            if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and isinstance(z, torch.Tensor)):
                raise ValueError('"x", "y", "z" should be torch.Tensor')
                
            if not (x.shape == y.shape == z.shape):
                raise ValueError('"x", "y", "z" should have same size')
                
            # Calculate level set
            loc = [x, y, z]
            level = torch.full_like(x, float('-inf'))
            
            for v in Axis.elems():
                level = torch.maximum(level, torch.abs(loc[v.value] - location[v.value]))
                
            return -level
        def lsf_zv(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, force_draw: bool = False) -> torch.Tensor:
            """
                Modified level set function that ensures visibility when force_draw is True.
                
                Args:
                    x (torch.Tensor): x-coordinates
                    y (torch.Tensor): y-coordinates
                    z (torch.Tensor): z-coordinates
                    force_draw (bool, optional): Flag to force drawing. Defaults to False.
                    
                Returns:
                    torch.Tensor: Modified level set values
            """
            # Calculate original level set
            level = lsf(x, y, z)
            
            if force_draw:
                # Flatten level set for easier operations
                level_flat = level.flatten()
                
                # Check if all values are negative
                if torch.all(level_flat <= 0):
                    # Find maximum value
                    max_level = torch.max(level_flat)
                    
                    # Find second largest value (excluding the maximum)
                    mask = level_flat < max_level
                    if torch.any(mask):
                        second_max = torch.max(level_flat[mask])
                        
                        # Shift the function upward by subtracting second largest value
                        level = level - second_max
                    
            return level
            
        # Initialize primary grid locations
        lprim = [None] * Axis.count()
        for w in Axis.elems():
            lprim[w.value] = location[w.value]
            
        # Initialize parent class
        super().__init__(lprim, lsf_zv)
        
        # Store location
        self._location = location
        
    @property
    def location(self) -> torch.Tensor:
        """Get point location."""
        return self._location