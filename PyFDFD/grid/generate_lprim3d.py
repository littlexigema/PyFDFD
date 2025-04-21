from ..base import Axis,Sign,GT
from ..shape.Box import Box
from ..shape.Shape import Shape
from .generate_lprim1d_part import generate_lprim1d_part
from .complete_lprim1d import complete_lprim1d
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
        """动态生成网格，暂时不支持"""
        """代码已支持，2025/4/1"""
        # intervals = [[]]*Axis.count()  # 使用字典来存储 intervals
        intervals = [[] for _ in range(Axis.count())]  # 使用列表来存储 intervals
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
        lprim0 = [[] for _ in range(Axis.count())]
        ldual0 = [[] for _ in range(Axis.count())]

        for w in Axis.elems():
            interval = intervals[w][0]
            Nw = round(interval.L / interval.dl_max)
            lprim0[w] = lprim0[w] + torch.linspace(*interval.bound,Nw+1).tolist()

        for src in src_array:
            for w in Axis.elems():
                lprim = src.l[w,GT.PRIM]
                lprim = lprim[~torch.isnan(lprim)].tolist()
                ldual = src.l[w,GT.DUAL]
                ldual = ldual[~torch.isnan(ldual)].tolist()
                lprim0[w] = lprim0[w]+lprim#GT.prim
                ldual0[w] = ldual0[w]+ldual#GT.dual

        # For each Axis element, perform grid generation and exception handling
        
        # lprim_cell = {}#{w.value: None for w in Axis.elems()}  # to store the generated grids
        # Npml = {w: {Sign.N: 0, Sign.P: 0} for w in Axis.elems()}  # PML counts for each axis

        for w in range(Axis.count()):
            try:
                intervals = [[]for _ in Axis.elems()]
                lprim_part = generate_lprim1d_part(domain.interval[w], Lpml[w, :], intervals[w], lprim0[w], ldual0[w])
                lprim = complete_lprim1d(lprim_part)
            except Exception as err1:
                raise err1 
                try:
                    lprim_part = generate_lprim1d_part(domain.interval(w), Lpml[w, Sign.n], intervals[w], lprim0[w], [])
                    lprim = complete_lprim1d(lprim_part)
                    ldual = np.mean([lprim[:-1], lprim[1:]], axis=0)  # take average in column
                except Exception as err2:
                    raise Exception(f"Maxwell:gridGen: {w}-axis grid generation failed.") from err2
                
                if not np.array_equal(np.setdiff1d(ldual0[w], ldual), []):
                    raise Exception(f"Maxwell:gridGen: {w}-axis grid generation failed.") from err1
            
            Npml[w, Sign.N] = torch.sum(lprim < lprim[0] + Lpml[w, Sign.N])
            Npml[w, Sign.P] = torch.sum(lprim > lprim[-1] - Lpml[w, Sign.P])
            lprim_cell.append(lprim)
    # return 0,0
    return lprim_cell, Npml