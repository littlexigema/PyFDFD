from typing import List, Union
import torch
from ..base.Sign import Sign
from ..base.GT import GT
from ..shape.Interval import Interval

def generate_lprim1d_part(domain: Interval, 
                         Lpml: Union[float, List[float]], 
                         interval_array: List[Interval], 
                         lprim0_array: List[float], 
                         ldual0_array: List[float]) -> List[List[float]]:
    """
    Generate primary grid points in 1D with dynamic grid generation.
    
    Args:
        domain: Domain interval
        Lpml: PML thickness (scalar or [negative_side, positive_side])
        interval_array: Array of intervals
        lprim0_array: Initial primary grid points
        ldual0_array: Initial dual grid points
        
    Returns:
        List of lists containing grid point segments
    """
    # Check domain
    if not isinstance(domain, Interval):
        raise ValueError('"domain" should be instance of Interval')
    dl_max = domain.dl_max

    # Process Lpml
    if isinstance(Lpml, (int, float)):
        Lpml = [Lpml] * Sign.count()
    elif isinstance(Lpml,torch.Tensor):
        assert len(Lpml) == Sign.count(),RuntimeError('Lpml should be a list of length {}'.format(Sign.count()))
        Lpml = Lpml.tolist()
    else:
        raise ValueError('"Lpml" should be a list or Tensor')
    # Lpml = torch.tensor(Lpml)
    
    if not (torch.all(torch.tensor(Lpml) >= 0) and domain.L >= Lpml[Sign.N] + Lpml[Sign.P]):
        raise ValueError('Invalid PML configuration')

    # Collect boundary points
    b_min = domain.bound[Sign.N]
    b_max = domain.bound[Sign.P]
    b_pml_min = b_min + Lpml[Sign.N]
    b_pml_max = b_max - Lpml[Sign.P]
    
    # Add boundary points to lprim0_array
    lprim0_array = lprim0_array + [b_min, b_pml_min, b_pml_max, b_max]#边界和PML边界添加到primary网格上
    
    # Add interval points
    lprim_inter = []
    for interval in interval_array:
        lprim_inter.extend(interval.lprim)
    
    # Filter points within domain
    lprim_inter = torch.tensor(lprim_inter)
    if len(lprim_inter) > 0:
        is_in = (lprim_inter>b_min) * (lprim_inter<b_max)
        lprim0_array += lprim_inter[is_in].tolist()
    # is_in = torch.tensor([(x > b_min) and (x < b_max) for x in lprim_inter])#将在domain边界内的Interval加入主网格
    # lprim0_array.extend([x for x, valid in zip(lprim_inter, is_in) if valid])

    # Sort and remove duplicates with tolerance
    lprim0_array = torch.unique(torch.tensor(lprim0_array))
    
    isnot_equal_approx = lambda a,b:torch.abs(torch.tensor(a)-torch.tensor(b))>dl_max*1e-2
    # def isequal_approx(a: float, b: float) -> bool:
    #     return abs(a - b) < dl_max * 1e-8
    
    # Remove nearly equal points
    diff_lprim0_array = lprim0_array[1:] - lprim0_array[:-1]
    ind_unique = isnot_equal_approx(diff_lprim0_array,0)
    lprim0_array = lprim0_array[:-1][ind_unique].tolist() + [lprim0_array[-1].item()]
    lprim0_array = torch.tensor(lprim0_array)
    if not isnot_equal_approx(lprim0_array[-1],lprim0_array[-2]).item():
        lprim0_array = lprim0_array[:-1]
    # i = 0
    # while i < len(lprim0_array) - 1:#两点相距过近认为是同一点
    #     if isequal_approx(lprim0_array[i], lprim0_array[i + 1]):
    #         lprim0_array.pop(i + 1)
    #     else:
    #         i += 1

    # Verify boundaries
    assert lprim0_array[0]==b_min and lprim0_array[-1]==b_max   

    # Check for overlapping primary and dual grids
    ldual0_array = torch.unique(torch.tensor(ldual0_array))
    common = set(lprim0_array) & set(ldual0_array)
    if common:
        raise RuntimeError(f'Primary and dual grid share {common}')

    # Generate primary grid points around dual grid points
    dl_prim0_array = lprim0_array[1:] - lprim0_array[:-1]
    dl_dual0_array = ldual0_array[1:] - ldual0_array[:-1]
    # a = torch.cat([dl_dual0_array, torch.tensor([float('inf')])]).view(1,-1)
    # b = torch.cat([torch.tensor([float('inf')]), dl_dual0_array]).view(1,-1)
    # dl_dual0_array = torch.cat([a,b],dim = 0)
    dl_dual0_array = torch.minimum(
        torch.cat([dl_dual0_array, torch.tensor([float('inf')])]),
        torch.cat([torch.tensor([float('inf')]), dl_dual0_array])
    )

    # Generate points around dual grid points
    lprim_by_ldual0 = []
    for j, val in enumerate(ldual0_array):
        mask =lprim0_array < val
        ind = torch.nonzero(mask)[-1].item() if mask.any() else -1
        # ind = next((i for i, x in enumerate(lprim0_array) if x > val), len(lprim0_array)) - 1
        dl_min = min(dl_max, dl_dual0_array[j].item())
        if ind >= 0:
            dl_min = min(dl_min, dl_prim0_array[ind].item())
        
        # Check intervals
        for interval in interval_array:
            if interval.contains(val):
                dl_min = min(dl_min, interval.dl_max)
        
        # Add new points
        """
        将dualgrid +- dl/2的点添加到primary grid上
        """
        for offset in [-0.5, 0.5]:
            new_point = val.item() + offset * dl_min
            if b_min <= new_point <= b_max:
                lprim_by_ldual0.append(new_point)

    # Update and sort primary grid points
    lprim0_array = lprim0_array.tolist()
    lprim_by_ldual0 = torch.unique(torch.tensor(lprim_by_ldual0)).tolist()
    lprim0_array.extend(lprim_by_ldual0)
    lprim0_array = torch.unique(torch.tensor(lprim0_array))
    #For each interval between primary grid points, find the smallest dl suggested by intervals.
    lprim0_mid_array = (lprim0_array[1:] + lprim0_array[:-1])/2
    dl_prim0_array = lprim0_array[1:] - lprim0_array[:-1]

    n_prim0 = len(lprim0_array)
    dl_mid_array = torch.ones(n_prim0-1)*torch.nan
    for j, val in enumerate(lprim0_mid_array):
        mask =lprim0_array < val
        dl_min = min(dl_max, dl_prim0_array[j])
        # Check intervals
        for interval in interval_array:
            if interval.contains(val):
                dl_min = min(dl_min, interval.dl_max)
        dl_mid_array[j] = dl_min

    dl_boundary_array = torch.minimum(
        torch.cat([torch.tensor([float('inf')]),dl_mid_array]),
        torch.cat([dl_mid_array, torch.tensor([float('inf')])])
    )    
    # # Generate final grid segments
    # lprim_part_cell = []
    # prev = None
    val = lprim0_array[0].item()
    dl = dl_boundary_array[0].item()
    prev = [val ,val+dl]
    lprim_part_cell = [prev]
    for i in range(1,len(lprim0_array)):
        val = lprim0_array[i].item()
        dl = dl_boundary_array[i].item()
        # if dl==4.4199981689453125:
        #     print(True)
        if val == b_max:
            curr = [val - dl,val]
        else:
            curr = [val - dl, val, val + dl]

        if not (isnot_equal_approx(curr[0],prev[-2]).item() or isnot_equal_approx(curr[1],prev[-1]).item()):
            """
            完全重叠情况
            当前段的前两个点与前一段的最后两个点近似重合
            保留前一段全部点，只添加当前段的剩余点
            """
            curr = prev + curr[2:]
            lprim_part_cell = lprim_part_cell[:-1]
            lprim_part_cell.append(curr)
        elif not isnot_equal_approx(curr[0],prev[-1]).item():
            curr = prev + curr[1:]
            lprim_part_cell = lprim_part_cell[:-1]
            lprim_part_cell.append(curr)
        elif curr[0]<prev[-1] and curr[-1]> prev[-1]:
            overlap_start = prev[-1]
            overlap_end = curr[-1]
            # 合并重叠部分
            curr = prev + [x for x in curr if x > overlap_start]
            lprim_part_cell = lprim_part_cell[:-1]
            lprim_part_cell.append(curr)
        else:
            lprim_part_cell.append(dl_mid_array[i-1].item())
            lprim_part_cell.append(curr)
        prev = curr
        # if i == 0:
        #     curr = [val, val + dl]
        # elif i == len(lprim0_array) - 1:
        #     curr = [val - dl, val]
        # else:
        #     curr = [val - dl, val, val + dl]
            
        # if prev is None:
        #     prev = curr
        #     lprim_part_cell.append(prev)
        # else:
        #     if isequal_approx(curr[0], prev[-2]) and isequal_approx(curr[1], prev[-1]):
        #         prev.extend(curr[2:])
        #     elif isequal_approx(curr[0], prev[-1]):
        #         prev.extend(curr[1:])
        #     else:
        #         lprim_part_cell.append(curr)
        #         prev = curr

    return lprim_part_cell