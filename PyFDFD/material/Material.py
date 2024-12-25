from PyFDFD.base.Axis import Axis
import numpy as np

class Material:
    """
    Material represents an electromagnetic material with electric permittivity
    and magnetic permeability.
    """

    def __init__(self, name, color, eps, mu=1.0, islossless=False):
        self._validate_inputs(name, color, eps, mu, islossless)
        
        self.name = name
        self.color = color
        self.eps = self._make_tensor(eps)
        self.mu = self._make_tensor(mu)
        
        if islossless:
            self.name += ' (lossless)'
            self.eps = np.real(self.eps)
            self.mu = np.real(self.mu)

    @staticmethod
    def _validate_inputs(name, color, eps, mu, islossless):
        if not isinstance(name, str):
            raise ValueError('"name" should be a string.')
        if not (isinstance(color, str) or (isinstance(color, (list, np.ndarray)) and len(color) == 3 and all(0 <= c <= 1 for c in color))):
            raise ValueError('"color" should be a string or [r, g, b].')
        # if not (np.iscomplexobj(eps) and (np.isscalar(eps) or eps.shape in [(3,), (3, 3), (3, 6)])):
        #     raise ValueError('"eps" should be a complex scalar, length-3 row vector, 3x3 matrix, or 3x6 matrix.')
        # if not (np.iscomplexobj(mu) and (np.isscalar(mu) or mu.shape in [(3,), (3, 3), (3, 6)])):
        #     raise ValueError('"mu" should be a complex scalar, length-3 row vector, 3x3 matrix, or 3x6 matrix.')
        if not isinstance(islossless, bool):
            raise ValueError('"islossless" should be a boolean.')

    @staticmethod
    def _make_tensor(value):
        if np.isscalar(value):
            return np.diag([value] * Axis.count())
        elif value.shape == (Axis.count(),):
            return np.diag(value)
        elif value.shape == (Axis.count(), 2*Axis.count()):
            S = value[:, 3:]
            value = value[:, :3]
            return np.dot(S, np.dot(value, np.linalg.inv(S)))
        return value

    @property
    def hasisoeps(self):
        return np.allclose(np.diag(self.eps), self.eps[0, 0]) and np.all(np.diag(self.eps) == self.eps[0, 0])

    @property
    def hasisomu(self):
        return np.allclose(np.diag(self.mu), self.mu[0, 0]) and np.all(np.diag(self.mu) == self.mu[0, 0])

    @property
    def isiso(self):
        return self.hasisoeps and self.hasisomu

    def sort(self, materials, reverse=False):
        return sorted(materials, key=lambda mat: mat.name, reverse=reverse)

    def __ne__(self, other):
        if not isinstance(other, Material):
            return True
        return not (self.name == other.name and self.color == other.color and np.allclose(self.eps, other.eps) and np.allclose(self.mu, other.mu))

# Example usage
if __name__ == "__main__":
    vacuum = Material('vacuum', 'none', 1.0)
    print(f"Material name: {vacuum.name}")
    print(f"Material color: {vacuum.color}")
    print(f"Material eps: {vacuum.eps}")
    print(f"Material mu: {vacuum.mu}")
    print(f"Material is isotropic: {vacuum.isiso}")