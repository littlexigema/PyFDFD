import torch

class ShiftedSigmoid(torch.nn.Module):
    def __init__(self, shift=0.0):
        super(ShiftedSigmoid, self).__init__()
        self.shift = shift

    def forward(self, x):
        return torch.sigmoid(x + self.shift)

class MLP(torch.nn.Module):
    """
    可以传入一个list, list长度代表层数, list中的元素代表每层的神经元数量
    可以传入一个list, list长度代表层数, list中的元素代表每层的激活函数
    """
    def __init__(self, layer_sizes, activations):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        # self.dropout = torch.nn.Dropout(p=0.01)
        self.noise_std = 0.0001
        # 展平嵌套列表
        layer_sizes = self.flatten_list(layer_sizes)
        activations = self.flatten_list(activations)
        for i in range(len(layer_sizes)-1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1], dtype=torch.float64))
            # self.layers.append(torch.nn.BatchNorm1d(num_features=layer_sizes[i+1], dtype=torch.float64))
            if activations[i] =='relu':
                self.layers.append(torch.nn.ReLU())
            elif activations[i] == 'tanh':
                self.layers.append(torch.nn.Tanh())
            elif activations[i] == 'leaky_relu':
                self.layers.append(torch.nn.LeakyReLU())
            elif activations[i] == 'sigmoid':
                self.layers.append(torch.nn.Sigmoid())
            elif activations[i] == 'linear':
                pass
            elif activations[i] == 'shifted_sigmoid':
                self.layers.append(ShiftedSigmoid(shift=-.1))
            elif activations[i] == 'silu':
                self.layers.append(torch.nn.SiLU())
            elif activations[i] == 'elu':
                self.layers.append(torch.nn.ELU())
            else:
                raise ValueError('Unsupported activation function')
        # torch.nn.init.zeros_(self.layers[-1].weight)
        # torch.nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # # 只在倒数第三层前添加 Dropout 和噪声
            # if i < len(self.layers) - 2:
            #     # x = self.dropout(x)  # 添加 Dropout
            #     noise = torch.randn_like(x) * self.noise_std  # 生成噪声
            #     x += noise  # 添加噪声
        # x = (x + 1)/2 # 如果是tanh激活函数的话
        return x
    
    def flatten_list(self, nested_list):
        flattened = []
        for item in nested_list:
            if isinstance(item, list):
                # 如果是子列表，递归调用 flatten_list
                flattened.extend(self.flatten_list(item))
            else:
                # 如果不是子列表，直接添加到结果中
                flattened.append(item)
        return flattened

if __name__ == '__main__':
    model = MLP([2, [3]*3, 1], [['relu']*3, 'relu'])
    # 展示网络结构
    print(model)
    