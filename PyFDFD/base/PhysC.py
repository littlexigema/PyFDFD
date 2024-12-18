import math

class PhysC:
    """
    PhysC is the class defining fundamental physical constants.
    """

    # Fundamental constants (class-level attributes)
    c0 = 2.99792458e8  # Speed of light in m/s
    mu0 = 4 * math.pi * 1e-7  # Permeability of free space in H/m
    eps0 = 1 / (c0**2 * mu0)  # Permittivity of free space in F/m
    eta0 = math.sqrt(mu0 / eps0)  # Impedance of free space in Ohms
    h = 4.135667662e-15  # Planck constant in eV·s
    hbar = h / (2 * math.pi)  # Reduced Planck constant in eV·s
