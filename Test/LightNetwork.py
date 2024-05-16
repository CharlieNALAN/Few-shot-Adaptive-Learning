import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个基本的卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 定义包含跳跃连接的网络
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.layer1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.layer2 = ConvBlock(64, 64, kernel_size=3)
        self.layer3 = ConvBlock(64, 128, kernel_size=3)
        self.layer4 = ConvBlock(128, 256, kernel_size=3)
        self.layer5 = ConvBlock(256, 512, kernel_size=3)
        self.layer6 = ConvBlock(512, 192, kernel_size=3)

        self.ca = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        # 跳跃连接
        identity = x
        out = self.ca(x)
        out += identity

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out


# 创建模型实例并打印
model = MyNetwork()
print(model)

# 测试输入
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(output)
