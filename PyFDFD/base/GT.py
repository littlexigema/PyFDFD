from enum import Enum

class GT(Enum):
    """
    GT is the enumeration class for the types of grids (PRIMary and dual).
    """
    PRIM = (0, 'primary')
    DUAL = (1, 'dual')

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
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
    def elems(ind=None):
        """
        Returns all elements of the GT enumeration.
        """
        elems = list(GT)
        if ind is not None:
            return elems[ind]
        return elems

    def alter(self):
        """
        Alters between GT.PRIM and GT.dual.
        """
        if self == GT.PRIM:
            return GT.DUAL
        elif self == GT.DUAL:
            return GT.PRIM
        return None

    @staticmethod
    def count():
        """
        Returns the count of GT elements.
        """
        return len(GT.elems())

# Example usage
if __name__ == "__main__":
    lst = ['primary', 'dual']
    print(lst[GT.PRIM])  # 自动转换为整数
    print(GT.PRIM)  # 输出: PRIM (PRIMary)
    print(int(GT.PRIM))  # 输出: 0
    print(repr(GT.PRIM))  # 输出: PRIM (PRIMary)
    print(GT.elems())  # 输出: [<GT.PRIM: 0>, <GT.dual: 1>]
    for gt in GT.elems():
        print(f"GT.{gt.name} corresponds to {gt.value}")

    # Test cyclic permutation
    print(GT.PRIM.alter())  # 输出: dual (dual)