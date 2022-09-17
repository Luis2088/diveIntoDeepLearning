import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

X = torch.rand(2, 20)


# print(net(X))


class MLP(nn.Module):
    # 用模型参数声明层。
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化

        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    # 定义模型的向前传播
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


# net = MLP()
# print(net(X))  # 这里全写应该是net.forward(X)


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # module是Module子类的一个实例。把它保存在'Module'类的成员变量_modules中
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict 保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))


# 在前向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数，因此在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


# net = FixedHiddenMLP()
# print(net(X))

# 混合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))
