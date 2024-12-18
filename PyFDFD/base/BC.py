from enum import Enum

class BC(Enum):
    """
    BC is an enumeration class for boundary conditions.
    """
    E = 'PEC'  # Tangential component of E-field = 0
    M = 'PMC'  # Normal component of E-field = 0
    P = 'periodic'  # Periodic boundary condition

    @staticmethod
    def elems():
        """
        Returns all the elements of the BC enumeration.
        """
        return list(BC)

    @staticmethod
    def count():
        """
        Returns the count of BC elements.
        """
        return len(BC.elems())

# Example usage
if __name__ == "__main__":
    print(f"# of instances of BC: {BC.count()}")
    for bc in BC.elems():
        print(f"BC.{bc.name} corresponds to {bc.value}")
