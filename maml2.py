import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.nn.functional as F

from dataset import TongueData


def create_resnet18(num_classes=4):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # 替换最后的全连接层，适应四分类任务
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


class MAML:
    def __init__(self, model):
        self.model = model
        self.meta_optimizer = optim.Adam(model.parameters(), lr=0.001)  # 元优化器

    def inner_update(self, model, data, lr_inner=0.01, num_updates=1):
        optimizer = optim.SGD(model.parameters(), lr=lr_inner)
        x, y = data

        for _ in range(num_updates):
            loss = nn.MSELoss()(model(x), y)  # 损失计算
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def meta_update(self, tasks, iterations=100, num_updates=1):
        for _ in range(iterations):
            for task in tasks:
                data = task
                cloned_model = self.clone_model()  # 克隆模型
                self.inner_update(cloned_model, data, lr_inner=0.01, num_updates=num_updates)  # 内部更新

                for p, cloned_p in zip(self.model.parameters(), cloned_model.parameters()):
                    # 更新原始模型的参数
                    if p.grad is None:
                        p.grad = cloned_p.grad.clone()
                    else:
                        p.grad += cloned_p.grad.clone()
                self.meta_optimizer.step()

    def clone_model(self):
        # 克隆模型的参数
        model_clone = create_resnet18(num_classes=4)
        model_clone.load_state_dict(self.model.state_dict())
        return model_clone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_resnet18(num_classes=4)
