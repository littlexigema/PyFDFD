import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 将项目根目录添加到 sys.path 中
sys.path.append(project_root)
from PyFDFD.material.Material import Material
from PyFDFD.material.NPY import NPY
from PyFDFD.shape.Shape import Shape
from PyFDFD.shape.Box import Box
import numpy as np

class EMObjectItem:
    """
    EMObjectItem is a helper class to store shape and material.
    """
    def __init__(self, shape, material):
        self.shape = shape
        self.material = material

class EMObject:
    """
    EMObject is the combination of Shape and Material.
    """

    def __init__(self, shape, material):
        if shape is not None and material is not None:
            self._validate_inputs(shape, material)
            
            self.obj = [EMObjectItem(shape[i], material) for i in range(len(shape))]
            # self.shape = shape
            # self.material = material

    @staticmethod
    def _validate_inputs(shape, material):
        if not isinstance(shape, list) or not all(isinstance(s, Shape) for s in shape):
            raise ValueError('"shape" should be a list of Shape instances.')
        if not isinstance(material, (Material,NPY)):
            raise ValueError('"material" should be an instance of Material.')

    def assign_eps_mu(self):
        # This function could assign eps and mu, but there are too many subpixel
        # smoothing schemes for eps and mu. Supporting all those schemes would
        # make EMObject very heavyweight. A decision was made to take out those
        # schemes from EMObject to leave it lightweight.
        pass

# Example usage
if __name__ == "__main__":
    # Assuming Shape and Material classes are defined elsewhere
    # shape1 = Shape()  # Replace with actual Shape instance
    # shape2 = Shape()  # Replace with actual Shape instance
    shape = Box(np.array([[-5, 5],[-10, 10], [0, 1]]), 0.1)
    material = Material('vacuum', 'none', 1.0)

    em_object = EMObject([shape], material)
    print(f"EMObject shapes: {[obj.shape for obj in em_object.obj]}")