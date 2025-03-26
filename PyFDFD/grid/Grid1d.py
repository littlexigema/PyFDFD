from PyFDFD.base.Axis import Axis
# from PyFDFD.base.PhysC import PhysC
# from PyFDFD.base.PhysQ import PhysQ
from PyFDFD.base.PhysUnit import PhysUnit,PhysC,PhysQ
from PyFDFD.base.BC import BC
from PyFDFD.base.Sign import Sign
from PyFDFD.base.GT import GT
# PhysQ,PhysUnit,BC,Sign,GT
import numpy as np
import torch

class Grid1d:
    """
    Grid1d contains all the information related to the staggered grid
    in a single Cartesian axis. It does not have physical quantities
    dependent on frequencies, e.g., omega, eps, mu, and PML s-factors.
    """

    def __init__(self, axis, unit, lprim_array, Npml_array, bc):
        assert isinstance(axis, Axis), '"axis" should be an instance of Axis.'
        self.axis = axis

        assert isinstance(unit, PhysUnit), '"unit" should be an instance of PhysUnit.'
        self.unit = unit
        self.unitvalue = unit.value(PhysQ.L)

        # if isinstance(Npml_array, np.ndarray):
        #     assert Npml_array.numel() == Sign.count(), \
        #     f'"Npml" should be a length-{Sign.count()} row vector with integer elements.'
        if isinstance(Npml_array, torch.Tensor):
            assert Npml_array.numel() == Sign.count(), \
            f'"Npml" should be a length-{Sign.count()} row vector with integer elements.'
        else:
            raise TypeError('"Npml_array" should be either a numpy array or a torch tensor.')
        self.Npml = Npml_array

        assert isinstance(bc, BC), '"bc" should be an instance of BC.'
        self.bc = bc

        assert isinstance(lprim_array, torch.Tensor) and len(lprim_array.shape) == 1, \
            '"lprim_array" should be a row vector with real elements.'

        # Set N and L.
        self.lprim = lprim_array
        self.N = len(self.lprim) - 1  # Number of grid cells
        self.L = self.lprim[-1] - self.lprim[0]

        # Set loc and dl.
        self.ldual = torch.empty(self.N + 1)
        self.ldual[1:] = (self.lprim[:-1] + self.lprim[1:]) / 2

        if self.bc == BC.P:
            self.ldual[0] = self.ldual[-1] - (self.lprim[-1] - self.lprim[0])
            self.ldual_ext = self.ldual[1] + (self.lprim[-1] - self.lprim[0])
        else:
            self.ldual[0] = self.lprim[0] - (self.ldual[1] - self.lprim[0])
            self.ldual_ext = self.lprim[-1] + (self.lprim[-1] - self.ldual[-1])

        self.l = [self.lprim[:-1], self.ldual[1:]]
        self.lghost = [self.lprim[-1].item(), self.ldual[0].item()]
        self.dl = [torch.diff(self.ldual), torch.diff(self.lprim)]

        # Set lpml, Lpml, and center.
        self.lpml = [self.lprim[self.Npml[Sign.N]].item(), self.lprim[-self.Npml[Sign.P] - 1].item()]#prim网格除去pml的两端的index位置
        self.Lpml = [(self.lpml[Sign.N] - self.lprim[0]).item(), (self.lprim[-1] - self.lpml[Sign.P]).item()]#记录pml的厚度
        self.center = torch.mean(torch.tensor(self.lpml))

        # Initialize kBloch
        self.kBloch = 0

    @property
    def lg(self):
        return [
            torch.append(self.l[GT.PRIM], self.lghost[GT.PRIM]),
            torch.insert(self.l[GT.DUAL], 0, self.lghost[GT.DUAL])
        ]

    @property
    def lall(self):
        return [
            self.lg[GT.PRIM],
            torch.append(self.lg[GT.DUAL], self.ldual_ext)
        ]

    @property
    def bound(self):
        return [self.lall[GT.PRIM][0], self.lall[GT.PRIM][-1]]

    def set_kBloch(self, blochSrc): 
        """
        如果需要使用这个函数,check the type of blochSrc
        """
        # raise RuntimeError('the type of blochSrc is :{}'.format(type(blochSrc)))
        # assert isinstance(blochSrc, WithBloch), '"blochSrc" should be an instance of WithBloch.'
        self.kBloch = blochSrc.kBloch(self.axis)

    def contains(self, l):
        # assert np.issubdtype(l.dtype, np.floating), '"l" should be an array with real elements.'
        return (l >= self.lall[GT.PRIM][0]) & (l <= self.lall[GT.PRIM][-1])

    def bound_plot(self, withpml):
        return self.bound if withpml else self.lpml

    def lplot(self, g, withinterp, withpml):
        assert isinstance(g, GT), '"g" should be an instance of GT.'
        assert isinstance(withinterp, bool), '"withinterp" should be a boolean.'
        assert isinstance(withpml, bool), '"withpml" should be a boolean.'

        if g == GT.PRIM:
            lplot = self.lall[g]
        else:
            lplot = self.l[g]

        if not withpml:
            lplot = lplot[self.Npml[Sign.NEG]:-self.Npml[Sign.POS]]

        if g == GT.DUAL and withinterp:
            lbound = self.bound_plot(withpml)
            lplot = torch.concatenate([[lbound[0]], lplot, [lbound[-1]]])

        return lplot

    def lvoxelbound(self, g, withpml):
        assert isinstance(g, GT), '"g" should be an instance of GT.'
        assert isinstance(withpml, bool), '"withpml" should be a boolean.'
        try:
            lvoxelbound = self.lall[alter(g)]
            if not withpml:
                lvoxelbound = lvoxelbound[self.Npml[Sign.NEG]:-self.Npml[Sign.POS]]
            return lvoxelbound
        except:
            raise RuntimeError('define the function alter()')
        
