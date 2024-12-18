from enum import Enum

class PhysQ(Enum):
    """
    PhysQ is the enumeration class representing physical quantities.
    """

    # Enumeration values with (name, symbol, SI unit)
    arbitrary = ("arbitrary", "", "AU")  # Arbitrary unit
    L = ("length", "L", "m")
    omega = ("frequency", "\u03c9", "rad/s")  # Unicode for omega: ω
    eps = ("permittivity", "\u03b5", "F/m")  # Unicode for epsilon: ε
    mu = ("permeability", "\u03bc", "H/m")  # Unicode for mu: μ
    E = ("E-field", "E", "V/m")
    H = ("H-field", "H", "A/m")
    J = ("electric current density", "J", "A/m^2")
    M = ("magnetic current density", "M", "V/m^2")
    I = ("electric current", "I", "A")
    V = ("electric voltage", "V", "V")
    S = ("Poynting vector", "S", "W/m^2")
    P = ("power", "P", "W")
    u = ("power density", "u", "W/m^3")

    def __init__(self, name, symbol, SI_unit):
        """
        Initialize the enumeration values.

        Args:
            name (str): The name of the physical quantity.
            symbol (str): The symbol of the physical quantity.
            SI_unit (str): The SI unit of the physical quantity.
        """
        self._name = name
        self.symbol = symbol
        self.SI_unit = SI_unit

    @staticmethod
    def elems(index=None):
        """
        Retrieve the elements of the enumeration.

        Args:
            index (int, optional): Index of the desired element. If None, returns all elements.

        Returns:
            list or PhysQ: The list of elements or a specific element.
        """
        elements = list(PhysQ)
        if index is not None:
            return elements[index]
        return elements

    @staticmethod
    def count():
        """
        Returns the count of physical quantities.

        Returns:
            int: The number of physical quantities.
        """
        return len(PhysQ)

    def __repr__(self):
        """
        String representation for the enumeration.

        Returns:
            str: A string describing the enumeration.
        """
        return f"PhysQ(name='{self._name}', symbol='{self.symbol}', SI_unit='{self.SI_unit}')"
