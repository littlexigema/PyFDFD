import PhysQ,PhysC

class PhysUnit:
    """
    PhysUnit defines the units of the physical quantities used in MaxwellFDFD.
    """

    def __init__(self, L0):
        """
        Initializes the PhysUnit with a reference length L0.
        
        Args:
            L0 (float): The reference length in meters.
        """
        self.va = [float('nan')] * PhysQ.count  # Array to hold values of units
        self.va[PhysQ.arbitrary] = 1
        self.va[PhysQ.L] = L0  # Length in meters (L0-dependent)
        self.va[PhysQ.omega] = PhysC.c0 / self.va[PhysQ.L]  # Frequency in rad/s (L0-dependent)
        self.va[PhysQ.eps] = PhysC.eps0  # Permittivity in eps0
        self.va[PhysQ.mu] = PhysC.mu0  # Permeability in mu0
        self.va[PhysQ.E] = 1  # Electric field in V/m
        self.va[PhysQ.H] = self.va[PhysQ.E] / PhysC.eta0  # Magnetic field in A/m
        self.va[PhysQ.J] = self.va[PhysQ.H] / self.va[PhysQ.L]  # Current density in A/m^2 (L0-dependent)
        self.va[PhysQ.M] = self.va[PhysQ.E] / self.va[PhysQ.L]  # Magnetic current density in A/m^2 (L0-dependent)
        self.va[PhysQ.I] = self.va[PhysQ.J] * self.va[PhysQ.L] ** 2  # Electric current in Amperes (L0-dependent)
        self.va[PhysQ.V] = self.va[PhysQ.E] * self.va[PhysQ.L]  # Voltage in Volts (L0-dependent)
        self.va[PhysQ.S] = self.va[PhysQ.E] * self.va[PhysQ.H]  # Poynting vector in Watt/m^2
        self.va[PhysQ.P] = self.va[PhysQ.S] * self.va[PhysQ.L] ** 2  # Power in Watts (L0-dependent)
        self.va[PhysQ.u] = self.va[PhysQ.S] / self.va[PhysQ.L]  # Power density in Watt/m^3 (L0-dependent)

    def value(self, physQcell):
        """
        Computes the value of a physical quantity based on its unit and dimension.

        Args:
            physQcell (list or PhysQ): A single PhysQ instance or a list of pairs 
                                       [(PhysQ, int), (PhysQ, int), ...].

        Returns:
            float: The computed value in the desired units.
        """
        if isinstance(physQcell, PhysQ):
            physQcell = [(physQcell, 1)]

        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in physQcell), \
            '"physQcell" should be an instance of PhysQ or a list of tuples [(PhysQ, int), ...].'

        v0 = 1
        for physQ, physDim in physQcell:
            v0 *= self.va[physQ] ** physDim

        return v0
