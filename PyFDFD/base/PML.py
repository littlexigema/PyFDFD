from enum import Enum

class PML(Enum):
    """
    PML is an enumeration class for the kinds of PML.
    """
    SC = ('stretched-coordinate')
    U = ('uniaxial')

    def __new__(cls, description):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)  # Automatically assign an integer value based on the order of definition
        obj.description = description
        return obj

    def __str__(self):
        return f'{self.name} ({self.description})'

    def __repr__(self):
        return f'{self.name} ({self.description})'

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value

    @staticmethod
    def elems():
        """
        Returns all the elements of the PML enumeration.
        """
        return list(PML)

    @staticmethod
    def count():
        """
        Returns the count of PML elements.
        """
        return len(PML.elems())