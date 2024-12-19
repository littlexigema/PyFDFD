import numpy as np

class Interval:
    """
    Interval is a 1D version of Shape. It represents a 1D interval with defined bounds and grid properties.
    """

    def __init__(self, lprim_array, dl_max):
        """
        Initialize an Interval instance.

        Parameters:
        lprim_array (array-like): A non-empty row vector with real elements, used to generate the grid.
        dl_max (float): Maximum allowed grid spacing. Must be positive.
        """
        if not (isinstance(lprim_array, (list, np.ndarray)) and len(lprim_array) > 0):
            raise ValueError("'lprim_array' should be a non-empty list or numpy array with real elements.")

        self.lprim = np.unique(np.asarray(lprim_array))  # Sort and remove duplicates

        self.bound = [self.lprim.min(), self.lprim.max()]  # Set bounds

        if not (isinstance(dl_max, (int, float)) and dl_max > 0):
            raise ValueError("'dl_max' should be positive and real.")
        self.dl_max = dl_max

    def L(self):
        """
        Compute the length of the interval.

        Returns:
        float: Length of the interval.
        """
        return self.bound[1] - self.bound[0]

    def contains(self, val):
        """
        Check if the given values are within the interval bounds.

        Parameters:
        val (array-like): Array of real values to check.

        Returns:
        tuple:
            - truth (numpy.ndarray): Boolean array indicating whether each value is within the bounds.
            - distance (numpy.ndarray): Distance of each value from the nearest bound.
        """
        val = np.asarray(val)
        if not np.issubdtype(val.dtype, np.floating):
            raise ValueError("'val' should be an array with real elements.")

        bn, bp = self.bound  # Lower and upper bounds

        truth = (val >= bn) & (val <= bp)

        # distance = None
        # if truth.size > 0:
        distance = np.minimum(np.abs(val - bn), np.abs(val - bp))

        return truth, distance
