from PyFDFD.base.PhysUnit import PhysUnit
from PyFDFD.grid.Grid1d import Grid1d
from PyFDFD.base.Axis import Axis
from PyFDFD.base.GT import GT
import torch

class Grid3d:
    """
    Grid3d class to represent the 3D Yee grid structure.
    """

    def __init__(self, unit:PhysUnit, lprim_cell:list, Npml=0, bc=None):
        """
        Initialize the Grid3d object.

        Args:
            unit (PhysUnit): Instance of PhysUnit.
            lprim_cell (list): List of arrays defining the primary grid in each axis.
            Npml (int or list): Number of cells in the PML region.
            bc (BC or list): Boundary conditions for each axis.
        """
        self._validate_unit(unit)
        self._validate_lprim_cell(lprim_cell)
        Npml = self._expand_to_matrix(Npml, 3, 2)  # Assuming 3 axes and 2 signs.
        # bc = self._expand_to_row(int(bc), 3) 
        bc = [bc for _ in range(Axis.count())]

        # Initialize Grid1d components for each axis.
        self.comp = []
        for w in Axis.elems():  # Assuming axis indices are 0, 1, 2.
            self.comp.append(Grid1d(w, unit, lprim_cell[int(w)], Npml[int(w)], bc[int(w)]))

    @property
    def unit(self):
        return self.comp[0].unit

    @property
    def unitvalue(self):
        return self.comp[0].unitvalue

    @property
    def l(self):
        l = [[None]*GT.count() for _ in range(Axis.count())]
        for w in Axis.elems():
            for g in GT.elems():
                l[w][g] = self.comp[w].l[g]
        return l
        # return [[comp.l[g] for g in range(2)] for comp in self.comp]  # Assuming 2 grid types: primary and dual.

    @property
    def lg(self):
        return [[comp.lg[g] for g in range(2)] for comp in self.comp]

    @property
    def lall(self):
        return [[comp.lall[g] for g in range(2)] for comp in self.comp]

    @property
    def bound(self):
        return torch.tensor([comp.bound for comp in self.comp])

    @property
    def dl(self):
        return [[comp.dl[g] for g in range(2)] for comp in self.comp]

    @property
    def bc(self):
        return [comp.bc for comp in self.comp]

    @property
    def N(self):
        return torch.tensor([comp.N for comp in self.comp])

    @property
    def Ncell(self):
        return list(self.N)

    @property
    def Ntot(self):
        return torch.prod(self.N)

    @property
    def L(self):
        return torch.tensor([comp.L for comp in self.comp])

    @property
    def Npml(self):
        return torch.tensor([comp.Npml for comp in self.comp])

    @property
    def lpml(self):
        return torch.tensor([comp.lpml for comp in self.comp])

    @property
    def Lpml(self):
        return torch.tensor([comp.Lpml for comp in self.comp])

    @property
    def center(self):
        return torch.tensor([comp.center for comp in self.comp])

    def set_kBloch(self, plane_src):
        for comp in self.comp:
            comp.set_kBloch(plane_src)

    @property
    def kBloch(self):
        return torch.tensor([comp.kBloch for comp in self.comp])

    def contains(self, x, y, z):
        """
        Check if the given points are contained in the grid.

        Args:
            x (array): x-coordinates.
            y (array): y-coordinates.
            z (array): z-coordinates.

        Returns:
            array: Boolean array indicating whether each point is inside the grid.
        """
        loc = [x, y, z]
        truth = torch.ones_like(x, dtype=bool)
        for axis, comp in enumerate(self.comp):
            truth &= comp.contains(loc[axis])
        return truth

    def _validate_unit(self, unit):
        assert isinstance(unit, PhysUnit), '"unit" should be an instance of PhysUnit.'

    def _validate_lprim_cell(self, lprim_cell):
        assert isinstance(lprim_cell, list) and len(lprim_cell) == 3, \
            '"lprim_cell" should be a list of 3 arrays.'

    def _expand_to_matrix(self, data, rows, cols):
        if (torch.is_tensor(data) and data.numel() == 1):
            return torch.full((rows, cols), data.item(),dtype = torch.long)
        elif isinstance(data, (list, torch.Tensor)) and data.ndim==1:
            assert len(data) == rows
            # if data.ndim == 1:
            data = data[:, None] if data.shape[0] == rows else data[None, :]
            return torch.tile(data, (1, cols),dtype = torch)
        elif isinstance(data, (torch.Tensor)) and data.shape == (rows, cols):
            return data.to(torch.long)
        else:
            raise ValueError(f'"data" should be scalar, list of length {rows}, or {rows}x{cols} array.')

    def _expand_to_row(self, data, length):
        if np.isscalar(data) or (torch.is_tensor(data) and data.numel() == 1):
            return np.full((1, length), data) if isinstance(data, (int, float)) else torch.full((1, length), data.item())
        elif isinstance(data, (list, np.ndarray, torch.Tensor)) and data.ndim==1:
            assert len(data) == length
            if isinstance(data, list):
                data = torch.tensor(data)
            return data
        else:
            raise ValueError(f'"data" should be scalar or list of length {length}.')
