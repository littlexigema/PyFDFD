from enum import Enum

class PML(Enum):
    # 定义枚举类型
    sc = "stretched-coordinate"
    u = "uniaxial"
    
    @staticmethod
    def elems(ind=None):
        # 获取所有枚举项
        elems = [PML.sc, PML.u]
        if ind is not None:  # 如果传入了索引，则返回对应的元素
            return elems[ind]
        return elems
    
    @staticmethod
    def count():
        # 返回枚举项的数量
        return len(PML.elems())

# 示例用法
print("所有枚举项:", PML.elems())  # 返回所有枚举项
print("指定索引1的枚举项:", PML.elems(1))  # 返回指定索引的枚举项
print("枚举项数量:", PML.count())  # 返回枚举项的数量
