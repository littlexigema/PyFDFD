from enum import Enum

class GT(Enum):
    """
    GT is the enumeration class for the types of grids (primary and dual).
    """
    prim = 'primary'
    dual = 'dual'

    @staticmethod
    def elems(ind=None):
        """
        Returns all elements of the GT enumeration.
        """
        elems = list(GT)
        if ind is not None:
            return elems[ind]
        return elems

    @staticmethod
    def count():
        """
        Returns the count of GT elements.
        """
        return len(GT.elems())

    def alter(self):
        """
        Alters between GT.prim and GT.dual.
        """
        if self == GT.prim:
            return GT.dual
        elif self == GT.dual:
            return GT.prim
        return None

# Example usage
if __name__ == "__main__":
    print(f"# of instances of GT: {GT.count()}")
    for grid in GT.elems():
        print(f"GT.{grid.name} corresponds to {grid.value}")
    
    # Test alter method
    grid = GT.prim
    print(f"Alter of {grid.name}: {grid.alter().name}")
    grid = GT.dual
    print(f"Alter of {grid.name}: {grid.alter().name}")
