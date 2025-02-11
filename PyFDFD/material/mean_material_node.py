import torch
from ..base.Axis import Axis
from ..base.GT import GT
from ..grid.Grid3d import Grid3d
from ..io.expand_node_array import expand_node_array

def mean_material_node(grid3d: Grid3d, gt: GT, material_node: list) -> list:
    """
    Calculate the mean material properties at the nodes.
    """
    if not isinstance(grid3d, Grid3d):
        raise ValueError('"grid3d" should be an instance of Grid3d.')
    if not isinstance(gt, GT):
        raise ValueError('"gt" should be an instance of GT.')
    if not (isinstance(material_node, list) and len(material_node) == Axis.count() and all(isinstance(m, torch.Tensor) and m.shape == torch.Size(grid3d.N.tolist()) for m in material_node)):
        raise ValueError(f'"material_node" should be a list of {Axis.count()} tensors with shape {grid3d.N} and complex elements.')

    for w in Axis.elems():
        material_node[w.value] = expand_node_array(grid3d, material_node[w.value])  # (Nx+2) x (Ny+2) x (Nz+2)

    if gt == GT.PRIM:
        # material parameters for fields on primary grid
        material_cell = arithmetic_mean_material_node(material_node)
    else:  # gt == GT.DUAL
        # material parameters for fields on dual grid
        material_cell = harmonic_mean_material_node(material_node)

    return material_cell

def arithmetic_mean_material_node(material_node: list) -> list:
    material_edge_cell = [None] * Axis.count()

    material_edge_cell[Axis.X.value] = (
        material_node[Axis.X.value][1:-1, 1:-1, 1:-1] +
        material_node[Axis.X.value][1:-1, :-2, 1:-1] +
        material_node[Axis.X.value][1:-1, 1:-1, :-2] +
        material_node[Axis.X.value][1:-1, :-2, :-2]
    ) / 4

    material_edge_cell[Axis.Y.value] = (
        material_node[Axis.Y.value][1:-1, 1:-1, 1:-1] +
        material_node[Axis.Y.value][1:-1, 1:-1, :-2] +
        material_node[Axis.Y.value][:-2, 1:-1, 1:-1] +
        material_node[Axis.Y.value][:-2, 1:-1, :-2]
    ) / 4

    material_edge_cell[Axis.Z.value] = (
        material_node[Axis.Z.value][1:-1, 1:-1, 1:-1] +
        material_node[Axis.Z.value][:-2, 1:-1, 1:-1] +
        material_node[Axis.Z.value][1:-1, :-2, 1:-1] +
        material_node[Axis.Z.value][:-2, :-2, 1:-1]
    ) / 4

    return material_edge_cell

def harmonic_mean_material_node(material_node: list) -> list:
    material_face_cell = [None] * Axis.count()

    material_face_cell[Axis.X.value] = 2.0 / (
        1.0 / material_node[Axis.X.value][:-2, 1:-1, 1:-1] +
        1.0 / material_node[Axis.X.value][1:-1, 1:-1, 1:-1]
    )

    material_face_cell[Axis.Y.value] = 2.0 / (
        1.0 / material_node[Axis.Y.value][1:-1, :-2, 1:-1] +
        1.0 / material_node[Axis.Y.value][1:-1, 1:-1, 1:-1]
    )

    material_face_cell[Axis.Z.value] = 2.0 / (
        1.0 / material_node[Axis.Z.value][1:-1, 1:-1, :-2] +
        1.0 / material_node[Axis.Z.value][1:-1, 1:-1, 1:-1]
    )

    return material_face_cell

# Example usage
# Assuming Grid3d, Axis, GT, and other necessary components are defined elsewhere
# grid3d = Grid3d(...)  # Define this based on your requirements
# gt = GT.prim  # or GT.dual
# material_node = [torch.randn(grid3d.N.tolist(), dtype=torch.cfloat) for _ in range(Axis.count())]  # Example material node array
# material_cell = mean_material_node(grid3d, gt, material_node)