# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 18:00:10 2020

@author: 17509
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

print(torch.__version__)
print(torchvision.__version__)
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

print(torch.__version__)
print(torchvision.__version__)

train_set = torchvision.datasets.FashionMNIST(
    root='E:\\allDataSet\\FashionMnist\\FashionMNIST\\FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]))


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = t
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = t.reshape(-1, 12 * 4 * 4)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# 创建网络实例
# print(torch.cuda.get_device_name(0))
network = Network()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# network = network.to(device)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)
time1 = time.time()
for epoch in range(20):
    total_loss = 0
    total_correct = 0
    for batch in train_loader:  # Get batch
        images, labels = batch
        preds = network(images)
        loss = F.cross_entropy(preds, labels)
        optimizer.zero_grad()  # 告诉优化器把梯度属性中权重的梯度归零，否则pytorch会累积梯度
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
    print("epoch :", epoch, "loss :", total_loss, "total_correct :", total_correct)

print("timegap :", time.time() - time1)
accuracy = total_correct / len(train_set)
print("accuracy:", accuracy)
