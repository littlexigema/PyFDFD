from PyFDFD.io.EMObject import EMObject
from PyFDFD.grid.Grid3d import Grid3d
from PyFDFD.shape.Box import Box
from PyFDFD.base.Axis import Axis
from PyFDFD.base.Sign import Sign
from PyFDFD.base.GT import GT
# import numpy as np

def assign_material_node(grid3d:Grid3d, object_array, eps_node_cell, mu_node_cell):
    if not isinstance(grid3d, Grid3d):
        raise ValueError('"grid3d" should be an instance of Grid3d.')
    if not isinstance(object_array, list) or not all(isinstance(obj, EMObject) for obj in object_array):
        raise ValueError('"object_array" should be a list of instances of EMObject.')

    if eps_node_cell is None:
        eps_node_cell = [np.full(grid3d.N, np.nan, dtype=complex) for _ in range(Axis.count())]
    if not (isinstance(eps_node_cell, list) and len(eps_node_cell) == Axis.count() and all(isinstance(e, np.ndarray) and e.shape == grid3d.N for e in eps_node_cell)):
        raise ValueError(f'"eps_node_cell" should be a list of {Axis.count()} arrays with shape {grid3d.N} and complex elements.')

    if mu_node_cell is None:
        mu_node_cell = [np.full(grid3d.N, np.nan, dtype=complex) for _ in range(Axis.count())]
    if not (isinstance(mu_node_cell, list) and len(mu_node_cell) == Axis.count() and all(isinstance(m, np.ndarray) and m.shape == grid3d.N for m in mu_node_cell)):
        raise ValueError(f'"mu_node_cell" should be a list of {Axis.count()} arrays with shape {grid3d.N} and complex elements.')

    ldual = [None] * Axis.count()  # locations of cell centers
    for w in Axis.elems():
        ldual[w.value] = grid3d.l[w.value, GT.DUAL]  # grid3d.l rather than grid3d.lg

    ind = [None] * Axis.count()  # indices
    for obj in object_array:
        shape = obj.shape
        material = obj.material
        for w in Axis.elems():
            bn = shape.bound(w, Sign.N)
            bp = shape.bound(w, Sign.P)
            in_idx = np.searchsorted(ldual[w.value], bn, side='left')
            ip_idx = np.searchsorted(ldual[w.value], bp, side='right') - 1
            ind[w.value] = slice(in_idx, ip_idx + 1)

        if isinstance(shape, Box):
            for w in Axis.elems():
                eps_node_cell[w.value][tuple(ind)] = material.eps[w.value, w.value]
                mu_node_cell[w.value][tuple(ind)] = material.mu[w.value, w.value]
        else:  # shape is not a Box
            X, Y, Z = np.meshgrid(ldual[Axis.X.value][ind[Axis.X.value]],
                                ldual[Axis.Y.value][ind[Axis.Y.value]],
                                ldual[Axis.Z.value][ind[Axis.Z.value]], indexing='ij')
            is_in = shape.contains(X, Y, Z)
            ind_tf = np.zeros(grid3d.N, dtype=bool)  # logical indices
            ind_tf[tuple(ind)] = is_in

            for w in Axis.elems():
                eps_node_cell[w.value][ind_tf] = material.eps[w.value, w.value]
                mu_node_cell[w.value][ind_tf] = material.mu[w.value, w.value]

    return eps_node_cell, mu_node_cell