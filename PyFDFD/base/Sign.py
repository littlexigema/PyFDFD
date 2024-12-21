from enum import Enum

class Sign(Enum):
    """
    Sign is an enumeration class for negative and positive signs.
    """
    N = 'negative'
    P = 'positive'

    @staticmethod
    def elems(ind=None):
        """
        Returns all elements of the Sign enumeration.
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

    def alter(self):
        """
        Alters the sign: returns positive if negative, and negative if positive.
        """
        if self == Sign.N:
            return Sign.P
        elif self == Sign.P:
            return Sign.N
        return None

# Example usage
if __name__ == "__main__":
    print(Sign.N.value)
    # lst = [1,2,3]
    # for w in 
    # print(f"# of instances of Sign: {Sign.count()}")
    # for sign in Sign.elems():
    #     print(f"Sign.{sign.name} corresponds to {sign.value}")
    
    # # Test alter method
    # sign = Sign.N
    # print(f"Alter of {sign.name}: {sign.alter().name}")
    # sign = Sign.P
    # print(f"Alter of {sign.name}: {sign.alter().name}")
