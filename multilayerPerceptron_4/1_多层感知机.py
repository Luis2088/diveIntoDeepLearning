import torch
from d2l import torch as d2l

# 常见激活函数


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

# ReLU max(x,0)
y = torch.relu(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(6, 2.5))
# d2l.plt.show()

# ReLU的导数
y.backward(torch.ones_like(x), retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(6, 2.5))
# d2l.plt.show()

# sigmoid函数
y = torch.sigmoid(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
# d2l.plt.show()

# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
# d2l.plt.show()


# 双曲函数
y = torch.tanh(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
# d2l.plt.show()

# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
d2l.plt.show()