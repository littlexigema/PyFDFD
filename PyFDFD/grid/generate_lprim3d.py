import numpy as np
from ..base.Axis import Axis
from ..base.Sign import Sign
from ..shape.Box import Box
from ..shape.Shape import Shape
from ..shape.Interval import Interval
import torch

""""
function [lprim_cell, Npml] = generate_lprim3d(domain, Lpml, shape_array, src_array, withuniform=False)
"""
def generate_lprim3d(domain:Box, Lpml, shape_array=list, src_array=list, withuniform=False):
    assert len(Lpml)==Axis.count(),RuntimeError()
    Lpml = torch.tensor([Lpml for i in range(Sign.count())]).T
    lprim_cell = [[] for w in range(Axis.count())]#torch.empty(1,Axis.count())
    Npml = torch.ones(Axis.count(),Sign.count())*torch.nan
    # withuniform = True#for test
    if withuniform:
        for w in range(Axis.count()):
            dl_intended = domain.dl_max[w]
            # tmp = domain.L[w]
            # tmp = tmp/dl_intended
            Nw = round(domain.L[w]/dl_intended)
            lprim = torch.linspace(domain.bound[w,0],domain.bound[w,1],Nw+1)
            assert Nw>0
            dl_generate = (lprim[1]-lprim[0]).item()
            error_dl = (dl_generate-dl_intended)/dl_intended
            if abs(error_dl)>1e-9:
                print(f"Maxwell:gridGen: grid vertex separation {dl_generate} in generated uniform grid "
              f"significantly differs from intended separation {dl_intended}: error is {error_dl * 100:.2f} percent.")
            Npml[w,0] = (lprim<(lprim[0]+Lpml[w,0])).sum()#sign.n#区域添加PML厚度
            Npml[w,1] = (lprim>(lprim[-1]-Lpml[w,1])).sum()#sign.p
            lprim_cell[w] = lprim
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
        lprim0 = [[] for w in range(Axis.count())]
        ldual0 = [[] for w in range(Axis.count())]

        for src in src_array:
            for w in range(Axis.count()):
                lprim0[w].append(src.l[w, 0])#GT.prim
                ldual0[w].append(src.l[w, 1])#GT.dual

        # For each Axis element, perform grid generation and exception handling
        # lprim_cell = {w: None for w in Axis.elems()}  # to store the generated grids
        # Npml = {w: {Sign.n: 0, Sign.p: 0} for w in Axis.elems()}  # PML counts for each axis

        for w in range(Axis.count()):
            try:
                lprim_part = generate_lprim1d_part(domain.interval[w], Lpml[w, :], intervals[w], lprim0[w], ldual0[w])
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

def generate_lprim1d_part(domain: Interval, Lpml, interval_array, lprim0_array, ldual0_array):
    """
        % Check "Lpml".
        chkarg(istypeof(Lpml, 'real'), 'element of "Lpml" should be real.');
        chkarg(all(Lpml >= 0), 'element of "Lpml" should be positive.');
        chkarg(isexpandable2row(Lpml, Sign.count), ...
            '"Lpml" should be scalar or length-%d vector.', Sign.count);
        Lpml = expand2row(Lpml, Sign.count);
    """
    # Check "Lpml"
    if not torch.is_tensor(Lpml) or torch.is_complex(Lpml):
        raise ValueError('element of "Lpml" should be a real tensor.')
    if not torch.all(Lpml >= 0):
        raise ValueError('element of "Lpml" should be positive.')
    if Lpml.numel() != Sign.count():
        raise ValueError(f'"Lpml" should be scalar or length-{Sign.count()} vector.')
    Lpml = Lpml.expand(Sign.count())

    dl_max = domain.dl_max
    b_min = domain.bound[0]#Sign.N
    b_max = domain.bound[1]#Sign.P
    b_pml_min = b_min + Lpml[0]#Sign.N
    b_pml_max = b_max - Lpml[1]#Sign.P
    lprim0_array = np.unique(np.concatenate([lprim0_array, [b_min, b_pml_min, b_pml_max, b_max]]))

    lprim_inter = np.concatenate([interval.lprim for interval in interval_array])
    is_in = (lprim_inter > b_min) & (lprim_inter < b_max)
    lprim0_array = np.unique(np.concatenate([lprim0_array, lprim_inter[is_in]]))

    dlprim0_array = np.diff(lprim0_array)
    ind_unique = np.abs(dlprim0_array) >= dl_max * 1e-8
    lprim0_array = np.concatenate([lprim0_array[ind_unique], [lprim0_array[-1]]])
    if np.abs(lprim0_array[-1] - lprim0_array[-2]) < dl_max * 1e-8:
        lprim0_array = lprim0_array[:-1]

    assert lprim0_array[0] == b_min and lprim0_array[-1] == b_max

    ldual0_array = np.unique(ldual0_array)
    common = np.intersect1d(lprim0_array, ldual0_array)
    if common.size > 0:
        raise Exception(f"Maxwell:gridGen: primary and dual grid share {common}.")

    lprim_by_ldual0 = []
    for val in ldual0_array:
        ind = np.searchsorted(lprim0_array, val) - 1
        dl_min = min(dl_max, dlprim0_array[ind])
        for interval in interval_array:
            if interval.contains(val)[0]:
                dl_min = min(dl_min, interval.dl_max)
        lprim_new = [val - dl_min / 2, val + dl_min / 2]
        lprim_by_ldual0.extend([p for p in lprim_new if b_min <= p <= b_max])

    lprim0_array = np.unique(np.concatenate([lprim0_array, lprim_by_ldual0]))

    lprim0_mid_array = (lprim0_array[:-1] + lprim0_array[1:]) / 2
    dl_mid_array = np.full_like(lprim0_mid_array, dl_max)
    for j, val in enumerate(lprim0_mid_array):
        for interval in interval_array:
            if interval.contains(val)[0]:
                dl_mid_array[j] = min(dl_mid_array[j], interval.dl_max)

    dl_boundary_array = np.minimum(dl_mid_array, np.roll(dl_mid_array, 1))

    lprim_part_cell = [[lprim0_array[0], lprim0_array[0] + dl_boundary_array[0]]]
    for j in range(1, len(lprim0_array)):
        val = lprim0_array[j]
        dl = dl_boundary_array[j]
        if val == b_max:
            curr = [val - dl, val]
        else:
            curr = [val - dl, val, val + dl]
        prev = lprim_part_cell[-1]
        if np.abs(curr[0] - prev[-2]) < dl_max * 1e-8 and np.abs(curr[1] - prev[-1]) < dl_max * 1e-8:
            lprim_part_cell[-1].extend(curr[2:])
        elif np.abs(curr[0] - prev[-1]) < dl_max * 1e-8:
            lprim_part_cell[-1].extend(curr[1:])
        else:
            lprim_part_cell.append(curr)
    return lprim_part_cell

def complete_lprim1d(lprim_part_cell):
    rt = 1.9
    rmax = 2.0

    lprim_array = lprim_part_cell[0]
    for i in range(1, len(lprim_part_cell)):
        dl_n = lprim_array[-1] - lprim_array[-2]
        next_part = lprim_part_cell[i]
        dl_p = next_part[1] - next_part[0]
        gap = [lprim_array[-1], next_part[0]]
        dl_target = lprim_part_cell[i - 1][2] if len(lprim_part_cell[i - 1]) > 2 else dl_n
        filler = fill_targeted_geometric(dl_n, gap, dl_target, dl_p, rt, rmax)
        lprim_array = np.concatenate([lprim_array, filler[1:-1], next_part])

    dlprim_array = np.diff(lprim_array)
    if np.any(dlprim_array[1:] / dlprim_array[:-1] > rmax) or np.any(dlprim_array[1:] / dlprim_array[:-1] < 1 / rmax):
        raise Exception("Maxwell:gridGen: grid generation failed due to non-smooth grid vertex separations.")
    return lprim_array

def fill_targeted_geometric(dl_n, gap, dl_t, dl_p, rt, rmax):
    if dl_n == dl_p:
        return fill_targeted_geometric_sym(dl_n, gap, dl_t, rt, rmax)
    dl_max = max(dl_n, dl_p)
    dl_min = min(dl_n, dl_p)
    n = int(np.ceil(np.log(dl_max / dl_min) / np.log(rt)))
    r = (dl_max / dl_min) ** (1 / n)
    dl_array = dl_min * (r ** np.arange(1, n + 1))

    if dl_n < dl_p:
        filler_n = np.cumsum([gap[0]] + list(dl_array))
        gap_sym = [filler_n[-1], gap[1]]
        filler_sym = fill_targeted_geometric_sym(dl_p, gap_sym, dl_t, rt, rmax)
        return np.concatenate([filler_n[:-1], filler_sym])
    else:
        filler_p = np.cumsum([gap[1]] + list(-dl_array[::-1]))
        gap_sym = [gap[0], filler_p[0]]
        filler_sym = fill_targeted_geometric_sym(dl_n, gap_sym, dl_t, rt, rmax)
        return np.concatenate([filler_sym, filler_p[1:]])

def fill_targeted_geometric_sym(dl_sym, gap, dl_t, rt, rmax):
    if np.abs(dl_t - dl_sym) < 1e-8:
        return fill_constant(dl_sym, dl_t, gap, rt, rmax)
    dl_max = dl_t
    dl_min = dl_sym
    n = int(np.ceil(np.log(dl_max / dl_min) / np.log(rt)))
    r = (dl_max / dl_min) ** (1 / n)
    dl_array = dl_min * (r ** np.arange(1, n + 1))

    L_graded = np.sum(dl_array)
    if 2 * L_graded > gap[1] - gap[0]:
        n = int(np.ceil(np.log(dl_max / dl_min) / np.log(rt)))
        r = (dl_max / dl_min) ** (1 / n)
        dl_array = dl_min * (r ** np.arange(1, n + 1))
        L_graded = np.sum(dl_array)
        n_dl_max = int(np.floor((gap[1] - gap[0] - 2 * L_graded) / dl_max))
        dl_max_array = np.full(n_dl_max, dl_max)
        L_dl_max = np.sum(dl_max_array)
        n_dl_min = int(np.ceil((gap[1] - gap[0] - 2 * L_graded - L_dl_max) / 2 / dl_min))
        dl_min_array = np.full(n_dl_min, dl_min)
        L_dl_min = np.sum(dl_min_array)
        L_graded = (gap[1] - gap[0] - L_dl_max - 2 * L_dl_min) / 2
        r = (L_graded / dl_min) ** (1 / n)
        dl_array = dl_min * (r ** np.arange(1, n + 1))
        dl_filler = np.concatenate([dl_min_array, dl_array, dl_max_array, dl_array[::-1], dl_min_array])
    else:
        n = int(np.ceil(np.log(dl_max / dl_min) / np.log(rt)))
        r = (dl_max / dl_min) ** (1 / n)
        dl_array = dl_min * (r ** np.arange(1, n + 1))
        L_graded = np.sum(dl_array)
        n_dl_max = int(np.floor((gap[1] - gap[0] - 2 * L_graded) / dl_max))
        dl_max_array = np.full(n_dl_max, dl_max)
        L_dl_max = np.sum(dl_max_array)
        n_dl_min = int(np.ceil((gap[1] - gap[0] - 2 * L_graded - L_dl_max) / 2 / dl_min))
        dl_min_array = np.full(n_dl_min, dl_min)
        L_dl_min = np.sum(dl_min_array)
        L_graded = (gap[1] - gap[0] - L_dl_max - 2 * L_dl_min) / 2
        r = (L_graded / dl_min) ** (1 / n)
        dl_array = dl_min * (r ** np.arange(1, n + 1))
        dl_filler = np.concatenate([dl_min_array, dl_array, dl_max_array, dl_array[::-1], dl_min_array])
    return np.cumsum([gap[0]] + list(dl_filler))

def fill_constant(dl_min, dl_max, gap, rt, rmax):
    L = gap[1] - gap[0]
    numcells = [int(np.floor(L / dl_max)), int(np.ceil(L / dl_max))]
    dl = L / np.array(numcells)
    if not (is_smooth([dl_min, dl[0]], rt) or is_smooth([dl_min, dl[1]], rmax)):
        raise Exception(f"dl = {dl_min} is too small or dl = {dl_max} is too large for gap size = {L}.")
    numcells = numcells[1] if not is_smooth([dl_min, dl[0]], rmax) else numcells[0]
    return np.linspace(gap[0], gap[1], numcells + 1)

def is_smooth(dl_array, rt):
    return np.all(dl_array[1:] / dl_array[:-1] < rt) and np.all(dl_array[1:] / dl_array[:-1] > 1 / rt)