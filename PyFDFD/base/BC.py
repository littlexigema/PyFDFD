from enum import Enum

class BC(Enum):
    """
    BC is an enumeration class for boundary conditions.
    """
    E = (0, 'PEC: Tangential component of E-field = 0')
    M = (1, 'PMC: Normal component of E-field = 0')
    P = (2, 'Periodic: Periodic boundary condition')

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj

    def __str__(self):
        return f'{self.value} ({self.description})'

    def __repr__(self):
        return f'{self.value} ({self.description})'

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value

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
    lst = [1, 2, 3]
    print(lst[BC.P])  # 自动转换为整数
    # print(f"# of instances of BC: {BC.count()}")
    # for bc in BC.elems():
    #     print(f"BC.{bc.name} corresponds to {bc.value}")