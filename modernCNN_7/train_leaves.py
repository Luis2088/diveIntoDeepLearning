import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns

# 读取训练集，查看数据
labels_dataframe = pd.read_csv('../data/classify-leaves/train.csv')
print(labels_dataframe.iloc[0:4, :], type(labels_dataframe.iloc[0:4, :]), '\n================================\n')


# print(labels_dataframe.describe())


# 添加条形数量注释
def barw(ax):
    for p in ax.patches:
        val = p.get_width()
        x = p.get_x() + p.get_width()  # x-position
        y = p.get_y() + p.get_height() / 2  # y-position
        ax.annotate(val, (x, y), size=8)


# 可视化各个分类的数量
# ax0 = sns.countplot(y=labels_dataframe['label'], order=labels_dataframe['label'].value_counts().index)
# barw(ax0)
# plt.yticks([])
# plt.show()

# label排序
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)  # 176

# labels生成互相转化的字典
class_to_num = dict(zip(leaves_labels, range(0, n_classes)))  # {'abies_concolor': 0, 'abies_nordmanniana': 1 .......}
num_to_class = {v: k for k, v in class_to_num.items()}  # {0: 'abies_concolor', 1: 'abies_nordmanniana' ... }


# 继承dataset ,编写数据集类
class LeavesData(Dataset):
    def __init__(self, csv_path, img_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
               Args:
                   csv_path (string): csv 文件路径
                   img_path (string): 图像文件所在路径
                   mode (string): 训练模式还是测试模式
                   valid_ratio (float): 验证集比例
               """

        self.resize_height = resize_height
        self.resize_width = resize_width

        self.img_path = img_path
        self.mode = mode

        # 读取csv 文件
        self.data_info = pd.read_csv(csv_path, header=None)  # 去掉表头  形式: 第0行为表头, 第1开始为数据

        self.data_len = len(self.data_info.index) - 1  # 数据总长度   index返回的是列(序列)的总长度,包括了表头,故要减一
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            self.train_image_name = np.asarray(self.data_info.iloc[1:self.train_len + 1, 0])  # 返回图像文件名称的numpy数组
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len + 1, 1])  # 返回图像对应类型的numpy数组
            self.image_name_arr = self.train_image_name
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image_name = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_name_arr = self.valid_image_name
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image_name = np.asarray(self.data_info.iloc[1:, 0])
            self.image_name_arr = self.test_image_name

        self.real_len = len(self.image_name_arr)

        print('finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):  # 必须重写该方法才能返回数据迭代器
        # 获取对应的文件名
        image_name = self.image_name_arr[index]

        # 读取图像文件
        image_temp = Image.open(self.img_path + image_name)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # 训练模式对图像进行增强，随机水平翻转
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        image_temp = transform(image_temp)

        if self.mode == 'test':
            return image_temp
        else:
            # 得到图像的string label
            label = self.label_arr[index]
            # 对应的数字标签
            number_label = class_to_num[label]

            return image_temp, number_label

    def __len__(self):
        return self.real_len


train_path = '../data/classify-leaves/train.csv'
test_path = '../data/classify-leaves/test.csv'
img_path = '../data/classify-leaves/'

train_dataset = LeavesData(train_path, img_path, mode='train')
valid_dataset = LeavesData(train_path, img_path, mode='valid')
test_dataset = LeavesData(test_path, img_path, mode='test')
print(train_dataset.mode, train_dataset.__len__())
print(valid_dataset.mode, valid_dataset.__len__())
print(test_dataset.mode, test_dataset.__len__())

# def data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8, shuffle=False, num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=8, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=0)


# 展示数据
def img_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()  # 降维
    image = image.transpose(1, 2, 0)  # 将通道维 调换到最后
    image = image.clip(0, 1)

    return image


fig = plt.figure(figsize=(10, 6))
columns = 4
rows = 2
data_iter = iter(train_loader)
inputs, classes = data_iter.next()
for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    ax.set_title(num_to_class[int(classes[idx])])
    plt.imshow(img_convert(inputs[idx]))
plt.show()


# GPU测试
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device = get_device()
print(device)


# 是否冻住模型前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# resnet模型
def res_model(num_classes, feature_extract=False, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft


# 超参数
learning_rate = 3e-4
weight_decay = 1e-3
num_epoch = 50
model_path = './pre_res_model.ckpt'

# 初试化模型
model = res_model(n_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

best_acc = 0.0
# training
for epoch in range(num_epoch):

    # -----训练--------
    model.train()
    # record information
    train_loss = []
    train_accu = []

    # Iterate the training set by batches
    for X, y in tqdm(train_loader):  # X = images ，y = labels
        X = X.to(device)
        y = y.to(device)
        y_hat = model(X)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算当前批量的精确度
        acc = (y_hat.argmax(dim=-1) == y).float().mean()

        train_loss.append(loss.item())
        train_accu.append(acc)

    # 计算一个周期的损失和精确度
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accu) / len(train_accu)

    print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # -----验证--------
    model.eval()
    valid_loss = []
    valid_accu = []

    for X, y in tqdm(valid_loader):
        with torch.no_grad():
            y_hat = model(X.to(device))

        loss = criterion(y_hat, y.to(device))

        acc = (y_hat.argmax(dim=-1) == y.to(device)).float().mean()

        valid_loss.append(loss.item())
        valid_accu.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accu) / len(valid_accu)

    print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print('saving model with acc {:.3f}'.format(best_acc))

saveFileName = './submission.csv'

# predict
model = res_model(n_classes)
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

predictions = []

for X in tqdm(test_loader):
    with torch.no_grad():
        y_hat = model(X.to(device))

    predictions.extend(y_hat.argmax(dim=-1).cpu().numpy().tolist())

preds = []
for i in predictions:
    preds.append(num_to_class[i])

test_data = pd.read_csv(test_path)
test_data['label'] = pd.Series(preds)
submission = pd.concat([test_data['image'], test_data['label']], axis=1)
submission.to_csv(saveFileName, index=False)
