import numpy as np
from ..base.Axis import Axis
from .Interval import Interval

class Shape:
    def __init__(self, lprim_list, lsf, dl_max=None):
        """
        Initialize a Shape object.

        Args:
            lprim_list (list): A list of arrays representing primary grid planes.
            lsf (callable): A level set function that takes (x, y, z) as inputs.
            dl_max (float or list, optional): Maximum grid cell size.
        """
        self._validate_lprim_list(lprim_list)
        self.lsf = lsf
        
        if dl_max is None:
            dl_max = np.inf
        
        self._validate_dl_max(dl_max)
        self.dl_max = [dl_max] * Axis.count()
        
        self.interval = [Interval(lprim_list[w], self.dl_max[w]) for w in range(Axis.count())]
        self.n_subpxls = 10

    @staticmethod
    def _validate_lprim_list(lprim_list):
        if not all(isinstance(lprim, np.ndarray) for lprim in lprim_list):
            raise ValueError("Each element in lprim_list must be a numpy array.")

    @staticmethod
    def _validate_dl_max(dl_max):
        if hasattr(dl_max, "__len__"):
            if not all(d > 0 for d in dl_max):
                raise ValueError("Each element in dl_max must be positive.")
        elif dl_max <= 0:
            raise ValueError("dl_max must be positive.")

    # @staticmethod
    # def _create_interval(lprim, dl):
    #     return {
    #         'lprim': lprim,
    #         'bound': [lprim.min(), lprim.max()],
    #         'dl_max': dl
    #     }

    @property
    def lprim(self):
        return [interval['lprim'] for interval in self.interval]

    @property
    def bound(self):
        return np.array([[interval['bound'][0], interval['bound'][1]] for interval in self.interval])

    @property
    def cb_center(self):
        return np.mean(self.bound, axis=1)

    @property
    def L(self):
        return np.diff(self.bound, axis=1).flatten()

    # @property
    # def dl_max(self):
    #     self.dl_max

    def circumbox_contains(self, x, y, z):
        truth = np.ones_like(x, dtype=bool)
        coords = [x, y, z]

        for i, interval in enumerate(self.interval):
            truth &= (coords[i] >= interval['bound'][0]) & (coords[i] <= interval['bound'][1])

        return truth

    def contains(self, x, y, z):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        if not (x.shape == y.shape == z.shape):
            raise ValueError("x, y, and z must have the same shape.")

        return self.lsf(x, y, z) >= 0

    def smoothing_params(self, box):
        box = np.asarray(box)
        if box.shape != (len(self.interval), 2):
            raise ValueError(f"box must be of shape ({len(self.interval)}, 2).")

        dl_sub = (box[:, 1] - box[:, 0]) / self.n_subpxls
        l_probe = [np.linspace(b[0] - dl / 2, b[1] + dl / 2, self.n_subpxls + 2) for b, dl in zip(box, dl_sub)]

        X, Y, Z = np.meshgrid(*l_probe, indexing='ij')
        F = self.lsf(X, Y, Z)

        Fint = F[1:-1, 1:-1, 1:-1]
        is_contained = Fint > 0
        at_interface = Fint == 0
        rvol = is_contained.sum() + 0.5 * at_interface.sum()
        rvol /= self.n_subpxls ** 3

        Fx, Fy, Fz = np.gradient(F, *dl_sub)
        gradF = np.stack([Fx[1:-1, 1:-1, 1:-1].flatten(),
                          Fy[1:-1, 1:-1, 1:-1].flatten(),
                          Fz[1:-1, 1:-1, 1:-1].flatten()], axis=1)

        normF = np.linalg.norm(gradF, axis=1)
        gradF /= normF[:, None]

        ndir = -gradF.sum(axis=0) / np.linalg.norm(gradF.sum(axis=0))
        return rvol, ndir