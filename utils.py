import torch
import math

def round_complex_tensor(tensor, decimals=4):
    """
    对 complex tensor 的实部和虚部分别保留指定的小数位数。
    
    Args:
        tensor (torch.Tensor): 复数张量。
        decimals (int): 保留的小数位数。
    
    Returns:
        torch.Tensor: 保留指定小数位数的复数张量。
    """
    factor = 10 ** decimals
    real_part = torch.round(torch.real(tensor) * factor) / factor
    imag_part = torch.round(torch.imag(tensor) * factor) / factor
    return real_part + 1j * imag_part

def find_nearest_opposite_point(i: int, m: int, n: int) -> int:
    """
    输入：
      - i: 第一类圆上的点编号（1~4m）
      - m: 第一类点数是 4m
      - n: 第二类点数是 4n
    
    输出：
      - 与第一类第 i 个点正对面最近的第二类点编号（1~4n）
    """
    total_m = 4 * m
    total_n = 4 * n

    if not (1 <= i <= total_m):
        raise ValueError("第一类点的下标必须在 1 到 4m 之间")

    # 计算第一类第 i 个点的角度（弧度）
    theta_1 = (2 * math.pi * (i - 1)) / total_m
    # 对面角度（加 π）
    theta_opposite = (theta_1 + math.pi) % (2 * math.pi)

    # 找出哪个第二类点的角度最接近 theta_opposite
    min_diff = float('inf')
    closest_j = -1

    for j in range(1, total_n + 1):
        theta_2 = (2 * math.pi * (j - 1)) / total_n
        diff = abs(theta_2 - theta_opposite)
        # 考虑圆周角度距离
        diff = min(diff, 2 * math.pi - diff)
        if diff < min_diff:
            min_diff = diff
            closest_j = j

    return closest_j


def opposite_point(index: int, n: int) -> int:
    """
    圆上有 4n 个点（编号从 1 到 4n），返回与 index 正对面的最近点的编号。
    """
    total_points = 4 * n
    if not (1 <= index <= total_points):
        raise ValueError("下标必须在 1 到 4n 之间")

    # 转为 0-based 后加 2n，再模总数，最后转回 1-based
    opposite = (index - 1 + 2 * n) % total_points + 1
    return opposite

if __name__=="__main__":
    ans = opposite_point(index = 1,n=18)
    print(ans)
    ans = opposite_point(index = 37,n=18)
    print(ans)
    ans = opposite_point(index = 56,n=18)
    print(ans)
    ans = find_nearest_opposite_point(i = 10,m=9,n=18)
    print(ans)
    ans = find_nearest_opposite_point(i = 11,m=9,n=18)
    print(ans)