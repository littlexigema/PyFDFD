from enum import Enum

class FT(Enum):
    # 定义枚举类型
    e = "E"
    h = "H"
    
    @staticmethod
    def elems(ind=None):
        # 获取所有枚举项
        elems = [FT.e, FT.h]
        if ind is not None:  # 如果传入了索引，则返回对应的元素
            return elems[ind]
        return elems
    
    @staticmethod
    def count():
        # 返回枚举项的数量
        return len(FT.elems())

    def alter(self):
        # 交换 FT.e 和 FT.h 的方法
        if not isinstance(self, FT):
            raise TypeError('"this" should have FT as elements.')

        # 处理空情况
        if self is None:
            return None
        else:
            # 交换 E 和 H
            return FT.e if self == FT.h else FT.h

if __name__=="__main__":
    # 示例用法
    print("所有枚举项:", FT.elems())  # 返回所有枚举项
    print("指定索引1的枚举项:", FT.elems(1))  # 返回指定索引的枚举项
    print("枚举项数量:", FT.count())  # 返回枚举项的数量

    # 示例交换
    current_field = FT.e
    altered_field = current_field.alter()
    print(f"当前字段: {current_field}, 交换后的字段: {altered_field}")
