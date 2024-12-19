from .FT import FT
from .GT import GT

class EquationType:
    def __init__(self, f, ge):
        # 类型检查
        if not isinstance(f, list) or not all(isinstance(item, FT) for item in f):
            raise TypeError('"f" should be a list of FT instances.')
        
        if not isinstance(ge, GT):
            raise TypeError('"ge" should be an instance of GT.')

        # 设置属性
        self._f = f
        self._ge = ge
    
    @property
    def f(self):
        return self._f

    @property
    def ge(self):
        return self._ge

if __name__=="__main__":
    # 示例
    eq_type = EquationType([FT.e], GT.prim)
    print(eq_type.f)  # 获取 f 属性
    # eq_type.f = [FT.h, FT.e]  # 这将抛出错误，无法修改 f 属性
