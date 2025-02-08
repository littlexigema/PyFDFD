import torch
from .Shape import Shape

class Box(Shape):
    """
    Box is a subclass of Shape representing a 3D box aligned with Cartesian axes.

    Parameters:
    - bound: A 2D array [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    - dl_max: Maximum grid cell size allowed (optional)
    """

    def __init__(self, bound, dl_max=torch.inf):
        # Validate `bound`
        bound = torch.tensor(bound,dtype = torch.float32)
        assert bound.shape == torch.Size([3, 2]), '"bound" should be a 3x2 array.'
        s = (bound[:, 1] - bound[:, 0]) / 2  # Semi-sides
        assert torch.all(s >= 0), '"bound" should have lower bounds smaller than upper bounds in all axes.'
        c = torch.mean(bound, axis=1)  # Center

        # Define the level set function
        def lsf(x, y, z):
            x, y, z = torch.asarray(x), torch.asarray(y), torch.asarray(z)
            assert x.shape == y.shape == z.shape, '"x", "y", "z" should have the same shape.'
            loc = [x, y, z]
            level = -torch.inf * torch.ones_like(x)
            for v in range(3):  # Iterate over x, y, z axes
                level = torch.maximum(level, torch.abs(loc[v] - c[v]) / s[v])
            return 1 - level

        # Prepare primary grid plane (lprim)
        lprim = [bound[axis, :] for axis in range(3)]

        # Call the parent constructor
        super().__init__(lprim, lsf, dl_max)

# Example Usage
# shape = Box([[-100, 100], [-50, 50], [0, 20]])