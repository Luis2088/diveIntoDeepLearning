import torch
import torch.nn.functional as F
from torch import nn


# 定义不带参数的自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))

# 将层作为组件合并到更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())


# Y = net(torch.rand(4, 8))
# print(Y.mean())


# 定义带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


linear = MyLinear(5, 3)
print(linear.weight)

print(linear(torch.rand(2, 5)))

# 使用自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))



