import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.nn.functional as F
from torchvision.transforms import transforms

from dataset import TongueData


# weights=ResNet18_Weights.IMAGENET1K_V1

def create_resnet18(num_classes=4):
    model = resnet18()
    # 替换最后的全连接层，适应四分类任务
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


class MAML:
    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes
        self.meta_optimizer = optim.SGD(model.parameters(), lr=0.001)  # 元优化器

    def inner_update(self, model, data, lr_inner=0.001):
        optimizer = optim.SGD(model.parameters(), lr=lr_inner)  # 内部优化器
        for x, y in data:
            x = x.unsqueeze(0)
            y = torch.tensor([y], dtype=torch.long)
            loss = nn.CrossEntropyLoss()(model(x), y)  # 计算损失
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

    def meta_update(self, iterations=100, k_shot=10, n_way=4, q_query=15):

        for _ in range(iterations):
            support_set, query_set = dataset.sample_task(n_way, k_shot, q_query)
            cloned_model = self.clone_model()  # 克隆模型
            self.inner_update(cloned_model, support_set, lr_inner=0.01)  # 内部更新
            for p, cloned_p in zip(self.model.parameters(), cloned_model.parameters()):
                # 更新原始模型的参数
                if p.grad is None:
                    p.grad = cloned_p.grad.clone()  # 如果原始模型的参数没有梯度，则直接赋值
                else:
                    p.grad += cloned_p.grad.clone()  # 累加克隆模型的梯度
            self.meta_optimizer.step()  # 使用元优化器更新元模型的参数
            # query_loader = DataLoader(query_set, batch_size=q_query)
            # for x,y in query_loader:
            #     query_loss = nn.CrossEntropyLoss()(cloned_model(x), y)  # 查询集损失
            #     self.meta_optimizer.zero_grad()  # 清零元优化器梯度
            #     query_loss.backward()  # 反向传播查询集损失
            #     self.meta_optimizer.step()  # 更新元模型参数
            total_accuracy=0
            with torch.no_grad():
                model.eval()
                for data in loader:
                    imgs, labels,_ = data
                    # imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    print(outputs.argmax(1))
                    accuary = (outputs.argmax(1) == labels).sum()
                    total_accuracy += accuary.item()
            print("整体测试集预测正确率：{}".format(total_accuracy / len(dataset)))



    def clone_model(self):
        # 克隆模型的参数
        model_clone = create_resnet18(num_classes=4)
        model_clone.load_state_dict(self.model.state_dict())
        return model_clone


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = TongueData("./data", transform=transform)
loader = DataLoader(dataset, batch_size=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_resnet18(num_classes=4)
maml = MAML(model, 4)
maml.meta_update()
