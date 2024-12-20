from enum import Enum

class Axis(Enum):
    """
    Axis is an enumeration class representing the Cartesian axes x, y, z.
    """

    X = 'x'
    Y = 'y'
    Z = 'z'

    @staticmethod
    def elems():
        """
        Returns all the elements of the Axis enumeration.
        """
        return list(Axis)

    @staticmethod
    def count():
        """
        Returns the count of Axis elements.
        """
        return len(Axis.elems())

    def cycle(self):
        """
        Returns a cyclic permutation of [Axis.X, Axis.Y, Axis.Z] such that r == self.

        Returns:
            tuple: (p, q, r) where p and q are the remaining elements of the permutation.
        """
        elems = Axis.elems()
        idx = elems.index(self)
        r = elems[idx]
        p = elems[(idx + 1) % len(elems)]
        q = elems[(idx + 2) % len(elems)]
        return p, q, r

# Example usage
if __name__ == "__main__":
    print(f"# of instances of Axis: {Axis.count()}")
    for w in Axis.elems():
        print(f"The integer value of Axis.{w.name} is {w.value}")

    # Test cyclic permutation
    # p, q, r = Axis.Y.cycle()
    # print(f"The cyclic permutation of [X, Y, Z] beginning with Y is [{r.name}, {p.name}, {q.name}]")
