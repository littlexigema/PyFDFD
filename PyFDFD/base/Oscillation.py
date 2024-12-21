import numpy as np
from PyFDFD.base.PhysUnit import PhysUnit,PhysC,PhysQ

class Oscillation:
    def __init__(self, wvlen, unit):
        # if not np.iscomplexobj(wvlen):
            # raise ValueError('"wvlen" should be complex.')
        self.wvlen = wvlen
        if not isinstance(unit, PhysUnit):
            raise ValueError('"unit" should be instance of PhysUnit.')
        self.unit = unit

    def in_L0(self):
        return self.wvlen

    def in_omega0(self):
        return 2 * np.pi / self.wvlen  # omega = angular_freq / unit.omega0

    def in_eV(self):
        return PhysC.h * PhysC.c0 / (self.wvlen * self.unit.value(PhysQ.L))  # E (in eV) = h c / lambda