import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import dataset

# 定义教师网络
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 224 * 224, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义学生网络
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 加载数据集
transform=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 以50%的概率水平翻转
        transforms.RandomRotation(30),  # 随机旋转±30度
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并调整为224x224
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机亮度和对比度
        # transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
dataset = dataset.TonguesDatasetInJson('../data',transform=transform)
trainloader = DataLoader(dataset, batch_size=20)
# 初始化网络
teacher_net = torchvision.models.resnet18()
student_net = torchvision.models.resnet18()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(student_net.parameters(), lr=0.001, momentum=0.9)


# 训练学生网络
def train_student(teacher_net, student_net, trainloader, criterion, optimizer, temperature=2.0, alpha=0.5):
    teacher_net.eval()
    student_net.train()

    for epoch in range(10):  # 训练10个epoch
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()

            # 教师网络的预测
            with torch.no_grad():
                teacher_outputs = teacher_net(inputs)

            # 学生网络的预测
            student_outputs = student_net(inputs)

            # 计算软标签损失
            soft_labels = nn.functional.softmax(teacher_outputs / temperature, dim=1)
            student_soft_outputs = nn.functional.log_softmax(student_outputs / temperature, dim=1)
            loss_soft = nn.functional.kl_div(student_soft_outputs, soft_labels, reduction='batchmean') * (
                        temperature ** 2)

            # 计算硬标签损失
            loss_hard = criterion(student_outputs, labels)

            # 总损失
            loss = alpha * loss_hard + (1 - alpha) * loss_soft
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(trainloader)}')


# 训练学生网络
train_student(teacher_net, student_net, trainloader, criterion, optimizer)

# 保存学生网络
torch.save(student_net.state_dict(), 'student_net.pth')
