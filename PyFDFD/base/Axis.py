from enum import Enum

class Axis(Enum):
    """
    Axis is an enumeration class representing the Cartesian axes x, y, z.
    """
    X = (0, 'x-axis')
    Y = (1, 'y-axis')
    Z = (2, 'z-axis')

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
        Returns all the elements of the Axis enumeration.
        """
        elems = [Axis.X, Axis.Y, Axis.Z]
        if ind is not None:
            elems = elems[ind]  
        return elems

    @staticmethod
    def count():
        """
        Returns the count of Axis elements.
        """
        return len(Axis.elems())
    def cycle(self):
        r = self#0,1,2
        p = Axis.elems((int(self)+1)%Axis.count())#1,2,0
        q = Axis.elems((int(self)+2)%Axis.count())#2,0,1
        return p,q,r


# Example usage
if __name__ == "__main__":
    lst = ['x', 'y', 'z']
    print(lst[Axis.X])  # 自动转换为整数
    # print(Axis.elems())
    for axis in Axis.elems():
        print(f"Axis.{axis.name} corresponds to {axis.value}")
    print(Axis.X)  # 输出: X (x-axis)
    print(int(Axis.X))  # 输出: 0
    print(repr(Axis.X))  # 输出: X (x-axis)
    print(Axis.X.value)
    # lst=[1,2,3]
    # for w in Axis.elems():
    #     print(lst[w])
    # print(f"# of instances of Axis: {Axis.count()}")
    # for w in Axis.elems():
    #     print(f"The integer value of Axis.{w.name} is {w.value}")

    # Test cyclic permutation
    # p, q, r = Axis.Y.cycle()
    # print(f"The cyclic permutation of [X, Y, Z] beginning with Y is [{r.name}, {p.name}, {q.name}]")
