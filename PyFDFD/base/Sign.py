from enum import Enum

class Sign(Enum):
    """
    Sign is an enumeration class for negative and positive signs.
    """
    N = (0, 'negative')
    P = (1, 'positive')

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
    def elems(ind = None):
        """
        Returns all the elements of the Sign enumeration.
        """
        elems = list(Sign)
        if ind is not None:
            return elems[ind]
        return elems

    @staticmethod
    def count():
        """
        Returns the count of Sign elements.
        """
        return len(Sign.elems())

# Example usage
if __name__ == "__main__":
    lst = ['negative', 'positive']
    print(lst[Sign.N])  # 自动转换为整数
    print(Sign.N)  # 输出: N (negative)
    print(int(Sign.N))  # 输出: 0
    print(repr(Sign.N))  # 输出: N (negative)
    print(Sign.elems())  # 输出: [<Sign.N: 0>, <Sign.P: 1>]
    for sign in Sign.elems():
        print(f"Sign.{sign.name} corresponds to {sign.value}")