import copy
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import random
import dataset
from torchvision.models import ResNet18_Weights

from dataset import TongueData



class ResNetMAML(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMAML, self).__init__()
        self.features = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features.fc = nn.Identity()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


def train_maml(model, dataset, epochs=50, n_way=4, k_shot=5, q_query=15):
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    loss_func.to(device)
    for epoch in range(epochs):
        total_loss = 0

        for _ in range(10):
            support_set, query_set = dataset.sample_task(n_way, k_shot, q_query)
            model.train()  # 确保模型在训练模式

            # 创建模型的副本，以便在内循环中进行参数更新
            model_copy = copy.deepcopy(model)
            optimizer_inner = optim.SGD(model_copy.parameters(), lr=0.001)  # 内循环使用的优化器

            # 内循环：仅在模型副本上进行
            for x, y in support_set:
                x = x.unsqueeze(0)
                y = torch.tensor([y], dtype=torch.long)
                x,y=x.to(device), y.to(device)
                pred = model_copy(x)
                loss = loss_func(pred, y)
                optimizer_inner.zero_grad()
                loss.backward()

                optimizer_inner.step()  # 在模型副本上应用梯度更新


            # 外循环：使用查询集评估模型副本，并更新原始模型
            optimizer.zero_grad()
            query_loss = torch.tensor(0.0, requires_grad=True)
            for x, y in query_set:
                x = x.unsqueeze(0)
                y = torch.tensor([y], dtype=torch.long)
                x,y=x.to(device), y.to(device)
                pred = model_copy(x)
                loss = loss_func(pred, y)
                query_loss = loss + query_loss  # 累积损失

            # query_loss /= len(query_set)
            query_loss.backward()  # 反向传播计算梯度

            optimizer.step()  # 更新原始模型的参数


            total_loss += query_loss.item()
        total_accuracy=0
        with torch.no_grad():
            model.eval()
            for data in test_loader:
                imgs,labels,_ = data
                imgs,labels = imgs.to(device),labels.to(device)
                output = model(imgs)
                print(output.argmax(1))
                accuary = (output.argmax(1) == labels).sum()
                total_accuracy += accuary.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / 100:.4f}")  # 打印每个epoch的平均损失
        print(f"Epoch {epoch + 1}, Accuracy: {total_accuracy / len(dataset):.4f}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNetMAML(num_classes=4)
transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
dataset = TongueData("./data", transform=transform)
total_size = len(dataset)
test_size = 60
train_size = total_size - test_size
train_dataset, test_dataset = random_split(dataset, [train_size,test_size])
test_loader = DataLoader(test_dataset, batch_size=10)
model.to(device)
train_maml(model, dataset)
