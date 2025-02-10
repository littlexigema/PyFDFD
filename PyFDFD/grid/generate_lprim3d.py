from ..base.Axis import Axis
from ..base.Sign import Sign
from ..shape.Box import Box
from ..shape.Shape import Shape
import numpy as np
import torch
""""
function [lprim_cell, Npml] = generate_lprim3d(domain, Lpml, shape_array, src_array, withuniform)
"""
def generate_lprim3d(domain:Box, Lpml, shape_array=list, src_array=list, withuniform=False):
    assert len(Lpml)==Axis.count(),RuntimeError()
    Lpml = torch.tensor([Lpml for i in range(Sign.count())]).T
    lprim_cell = []#torch.empty(1,Axis.count())
    Npml = torch.ones(Axis.count(),Sign.count())*torch.nan
    withuniform = True#for test
    if withuniform:
        for w in range(Axis.count()):
            dl_intended = domain.dl_max[w]
            # tmp = domain.L[w]
            # tmp = tmp/dl_intended
            Nw = torch.round(domain.L[w]/dl_intended).to(torch.long)
            lprim = torch.linspace(domain.bound[w,0].item(),domain.bound[w,1].item(),Nw+1)
            assert Nw>0
            dl_generate = lprim[1]-lprim[0]
            error_dl = (dl_generate-dl_intended)/dl_intended
            if abs(error_dl)>1e-9:
                print(f"Maxwell:gridGen: grid vertex separation {dl_generate} in generated uniform grid "
              f"significantly differs from intended separation {dl_intended}: error is {error_dl * 100:.2f} percent.")
            Npml[w,0] = (lprim<lprim[0]+Lpml[w,0]).sum()#sign.n
            Npml[w,1] = (lprim>lprim[-1]-Lpml[w,1]).sum()#sign.n
            lprim_cell.append(lprim)
    else:
        intervals = [[] for w in range(Axis.count())]  # 使用字典来存储 intervals

        for shape in shape_array:
            for w in Axis.elems():
                inters_w = intervals[w]  # initially empty
                inter_curr = shape.interval[w]
                is_new = True
                i = 0
                while is_new and i < len(inters_w):
                    is_new = is_new and inters_w[i] != inter_curr  # compare contents of two objects
                    i += 1
                
                # Keep only a new interval; many intervals can be the same, e.g., in a photonic crystal.
                if is_new:
                    intervals[w].append(inter_curr)

        # Initialize lprim0 and ldual0
        lprim0 = [[] for w in Axis.count()]
        ldual0 = [[] for w in Axis.count()]

        for src in src_array:
            for w in range(Axis.count()):
                lprim0[w].append(src.l[w, 0])#GT.prim
                ldual0[w].append(src.l[w, 1])#GT.dual

        # For each Axis element, perform grid generation and exception handling
        lprim_cell = {w: None for w in Axis.elems()}  # to store the generated grids
        Npml = {w: {Sign.n: 0, Sign.p: 0} for w in Axis.elems()}  # PML counts for each axis

        for w in range(Axis.count()):
            try:
                lprim_part = generate_lprim1d_part(domain.interval(w), Lpml[w, Sign.n], intervals[w], lprim0[w], ldual0[w])
                lprim = complete_lprim1d(lprim_part)
            except Exception as err1:
                try:
                    lprim_part = generate_lprim1d_part(domain.interval(w), Lpml[w, Sign.n], intervals[w], lprim0[w], [])
                    lprim = complete_lprim1d(lprim_part)
                    ldual = np.mean([lprim[:-1], lprim[1:]], axis=0)  # take average in column
                except Exception as err2:
                    raise Exception(f"Maxwell:gridGen: {w}-axis grid generation failed.") from err2
                
                if not np.array_equal(np.setdiff1d(ldual0[w], ldual), []):
                    raise Exception(f"Maxwell:gridGen: {w}-axis grid generation failed.") from err1
            
            Npml[w, Sign.n] = np.sum(lprim < lprim[0] + Lpml[w, Sign.n])
            Npml[w, Sign.p] = np.sum(lprim > lprim[-1] - Lpml[w, Sign.p])
            lprim_cell[w] = lprim
    # return 0,0
    return lprim_cell, Npml