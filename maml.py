import copy
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
import random


class TongueImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.labels = self._load_labels()

    def _load_labels(self):
        labels = {}
        for image_file in self.image_files:
            json_path = os.path.join(self.root_dir, image_file.replace('.jpg', '.json'))
            with open(json_path, 'r') as f:
                data = json.load(f)
                labels[image_file] = data['label']
        return labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        label = self.labels[self.image_files[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label

    def sample_task(self, n_way, k_shot, q_query):
        task_classes = np.random.choice(list(set(self.labels.values())), n_way, replace=False)
        support_images = []
        query_images = []

        for cls in task_classes:
            cls_images = [(img, lbl) for img, lbl in zip(self.image_files, self.labels.values()) if lbl == cls]
            cls_images = random.sample(cls_images, k_shot + q_query)
            support_images.extend([(os.path.join(self.root_dir, img), lbl) for img, lbl in cls_images[:k_shot]])
            query_images.extend([(os.path.join(self.root_dir, img), lbl) for img, lbl in cls_images[k_shot:]])

        support_set = [(self.transform(Image.open(img)), lbl) for img, lbl in support_images]
        query_set = [(self.transform(Image.open(img)), lbl) for img, lbl in query_images]

        return support_set, query_set


class ResNetMAML(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMAML, self).__init__()
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Identity()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


def train_maml(model, dataset, epochs=5, n_way=5, k_shot=5, q_query=15):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0

        for _ in range(100):  # 假设每个epoch有100个任务
            support_set, query_set = dataset.sample_task(n_way, k_shot, q_query)
            model.train()  # 确保模型在训练模式

            # 创建模型的副本，以便在内循环中进行参数更新
            model_copy = copy.deepcopy(model)
            optimizer_inner = optim.Adam(model_copy.parameters(), lr=0.01)  # 内循环使用的优化器

            # 内循环：仅在模型副本上进行
            for x, y in support_set:
                x = x.unsqueeze(0)
                y = torch.tensor([y]).long()
                pred = model_copy(x)
                loss = loss_func(pred, y)
                optimizer_inner.zero_grad()
                loss.backward()
                optimizer_inner.step()  # 在模型副本上应用梯度更新

            # 外循环：使用查询集评估模型副本，并更新原始模型
            optimizer.zero_grad()
            query_loss = 0
            for x, y in query_set:
                x = x.unsqueeze(0)
                y = torch.tensor([y]).long()
                with torch.no_grad():
                    pred = model_copy(x)
                    loss = loss_func(pred, y)
                    query_loss += loss.item()
            total_loss += query_loss / len(query_set)

            # 使用查询集的损失更新原始模型
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / 100:.4f}")


model = ResNetMAML(num_classes=10)
dataset = TongueImageDataset(
    root_dir='/dataset',  # TODO 待调整
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

train_maml(model, dataset)
