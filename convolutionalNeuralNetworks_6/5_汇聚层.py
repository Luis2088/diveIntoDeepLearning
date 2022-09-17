import torch
from torch import nn
from d2l import torch as d2l


# 实现汇聚层的前向传播
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y


# 测试pool2d
# X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# print(pool2d(X, (2, 2)))

# 汇聚层的填充和步幅
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)

# 默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同。
# pool2d = nn.MaxPool2d(3, stride=1)
# print(pool2d(X))

# 在处理多通道输入数据时，汇聚层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总。
# 这意味着汇聚层的输出通道数与输入通道数相同。

X = torch.cat((X, X + 1), 1)
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
