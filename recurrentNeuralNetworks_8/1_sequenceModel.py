import torch
from torch import nn
from d2l import torch as d2l

# 使用正弦函数和一些可加性噪声来生成序列数据,时间步为1,2,...1000
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

# 马尔科夫
# 将这个序列转换为模型的“特征－标签”（feature-label）对。如果拥有足够长的序列就丢弃这几项； 另一个方法是用零填充序列。
tau = 4  # 预测值的特征为该值的前tau个值！！！
features = torch.zeros((T - tau, tau))  # (样本数，特征)
for i in range(tau):
    features[:, i] = x[i:T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 64, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(tau, tau * 2), nn.ReLU(), nn.Linear(tau * 2, 1))
    net.apply(init_weights)
    return net


loss = nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch{epoch + 1},'
              f'loss:{d2l.evaluate_loss(net, train_iter, loss):f}')


net = get_net()
train(net, train_iter, loss, 15, 0.05)

onestep_preds = net(features)
d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()], 'time', 'x',
         legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

# k步预测
multistep_preds = torch.zeros(T)
print(multistep_preds.shape)
multistep_preds[:n_train + tau] = x[:n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1))) # 用预测的数值来预测

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(), multistep_preds[n_train + tau:].detach().numpy()],
         'time', 'x', legend=['data', '1-step', 'multistep preds'], xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
print(features.shape)

for i in range(tau):
    features[:, i] = x[i:i + T - tau - max_steps + 1]

for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1:T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps],
         'time', 'x', legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000], figsize=(6, 3))
d2l.plt.show()
