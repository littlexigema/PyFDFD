from ..base import Axis,Sign
import torch
""""
function [lprim_cell, Npml] = generate_lprim3d(domain, Lpml, shape_array, src_array, withuniform)
"""
def generate_lprim3d(domain, Lpml, shape_array, src_array, withuniform):
    assert len(Lpml)==Axis.count(),RuntimeError()
    Lpml = torch.tensor([Lpml for i in range(Sign.count())]).T
    return 0,0
    # return lprim_cell, Npml