import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                    nn.Linear(120, 84), nn.Sigmoid(),
                    nn.Linear(84, 10))

# 记载模型参数
net.load_state_dict(torch.load('leNet.params'))
device = d2l.try_gpu()
net.to(device)

def predict_ch3(net, test_iter, n=30):
    for X, y in test_iter:
        X = X.to(device)
        y = y.to(device)
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)).cpu(), num_rows=n // 10,
        num_cols=10, titles=titles[0:n])
    d2l.plt.show()


def main():
    predict_ch3(net, test_iter)


if __name__ == '__main__':
    main()
