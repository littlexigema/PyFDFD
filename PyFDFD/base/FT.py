from enum import Enum

class FT(Enum):
    """
    FT is the enumeration class for the typel of fields.
    """
    E = (0,'E')
    H = (1,'H')
    
    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj

    def __str__(self):
        return f'{self.name} ({self.description})'

    def __repr__(self):
        return f'{self.name} ({self.description})'

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value

    @staticmethod
    def elems(ind=None):
        # 获取所有枚举项
        elems = list(FT)
        if ind is not None:  # 如果传入了索引，则返回对应的元素
            return elems[ind]
        return elems
    
    @staticmethod
    def count():
        # 返回枚举项的数量
        return len(FT.elems())

    def alter(self):
        if self == FT.E:
            return FT.H
        elif self == FT.H:
            return FT.E
        return None

if __name__=="__main__":
    # 示例用法
    # print("所有枚举项:", FT.elems())  # 返回所有枚举项
    # print("指定索引1的枚举项:", FT.elems(1))  # 返回指定索引的枚举项
    # print("枚举项数量:", FT.count())  # 返回枚举项的数量
    lst=[a for a in range(10)]
    for ft in FT.elems():
        print(lst[ft])  # 自动转换为整数

    # 示例交换
    current_field = FT.E
    altered_field = current_field.alter()
    print(f"当前字段: {current_field}, 交换后的字段: {altered_field}")
