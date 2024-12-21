import numpy as np

class PhysQ:
    arbitrary = 0
    L = 1
    omega = 2
    eps = 3
    mu = 4
    E = 5
    H = 6
    J = 7
    M = 8
    I = 9
    V = 10
    S = 11
    P = 12
    u = 13

    @staticmethod
    def elems(ind=None):
        elems = [PhysQ.arbitrary, PhysQ.L, PhysQ.omega, PhysQ.eps, PhysQ.mu, PhysQ.E, PhysQ.H, PhysQ.J, PhysQ.M, PhysQ.I, PhysQ.V, PhysQ.S, PhysQ.P, PhysQ.u]
        if ind is not None:
            elems = [elems[i] for i in ind]
        return elems

    @staticmethod
    def count():
        return len(PhysQ.elems())

class PhysC:
    c0 = 2.99792458e8  # speed of light in vacuum (m/s)
    mu0 = 4 * np.pi * 1e-7  # permeability of free space (H/m)
    eps0 = 1 / (c0**2 * mu0)  # permittivity of free space (F/m)
    eta0 = np.sqrt(mu0 / eps0)  # impedance of free space (Ohm)
    h = 4.135667662e-15  # Planck constant (eV·s)
    hbar = h / (2 * np.pi)  # reduced Planck constant (eV·s)

class PhysUnit:
    def __init__(self, L0):
        self.va = np.full(PhysQ.count(), np.nan)
        self.va[PhysQ.arbitrary] = 1
        self.va[PhysQ.L] = L0  # length in m (L0-dependent)
        self.va[PhysQ.omega] = PhysC.c0 / self.va[PhysQ.L]  # frequency in rad/s (L0-dependent)
        self.va[PhysQ.eps] = PhysC.eps0  # permittivity in eps0
        self.va[PhysQ.mu] = PhysC.mu0  # permeability in mu0
        self.va[PhysQ.E] = 1  # E-field in V/m
        self.va[PhysQ.H] = self.va[PhysQ.E] / PhysC.eta0  # H-field in A/m
        self.va[PhysQ.J] = self.va[PhysQ.H] / self.va[PhysQ.L]  # electric current density in A/m^2 (L0-dependent)
        self.va[PhysQ.M] = self.va[PhysQ.E] / self.va[PhysQ.L]  # magnetic current density in A/m^2 (L0-dependent)
        self.va[PhysQ.I] = self.va[PhysQ.J] * self.va[PhysQ.L]**2  # electric current in Amperes (L0-dependent)
        self.va[PhysQ.V] = self.va[PhysQ.E] * self.va[PhysQ.L]  # voltage in Volts (L0-dependent)
        self.va[PhysQ.S] = self.va[PhysQ.E] * self.va[PhysQ.H]  # Poynting vector in Watt/m^2
        self.va[PhysQ.P] = self.va[PhysQ.S] * self.va[PhysQ.L]**2  # power in Watt (L0-dependent)
        self.va[PhysQ.u] = self.va[PhysQ.S] / self.va[PhysQ.L]  # power density in Watt/m^3 (L0-dependent)

    def value(self, physQcell):
        if isinstance(physQcell, int):
            physQcell = [(physQcell, 1)]
        elif not all(isinstance(item, tuple) and len(item) == 2 for item in physQcell):
            raise ValueError('"physQcell" should be instance of PhysQ, or list of tuples [(PhysQ, int), ...].')

        v0 = 1
        for physQ, physDim in physQcell:
            v0 *= self.va[physQ]**physDim
        return v0

    # Uncomment and implement these methods if needed
    # def SI2normal(self, v_SI, physQ):
    #     if not isinstance(physQ, int):
    #         raise ValueError('"physQ" should be instance of PhysQ.')
    #     return v_SI / self.va[physQ]

    # def normal2SI(self, v_normal, physQ):
    #     if not isinstance(physQ, int):
    #         raise ValueError('"physQ" should be instance of PhysQ.')
    #     return v_normal * self.va[physQ]