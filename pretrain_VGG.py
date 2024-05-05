import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights
from torchvision.transforms import transforms

from dataset import TongueData
transform = transforms.Compose([
    transforms.Resize((192, 144)),
    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    # transforms.RandomRotation(15),  # 图像随机旋转±15度
    transforms.ToTensor(),  # 转换为张量
])
root_dir = "./dataset/train"
dataset = TongueData(root_dir=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model_VGG = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model_VGG.classifier[6] = nn.Linear(in_features=4096, out_features=2)
# print(model_VGG)
# model_VGG=torchvision.models.resnet18(pretrained=True)
# model_VGG.fc = nn.Linear(in_features=512,out_features=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = torch.tensor([0.25, 1.0, 1.0, 1.0, 0.85], dtype=torch.float32)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
optimizer2 = torch.optim.Adam(model_VGG.parameters(),lr=0.001)
model_VGG.to(device)
loss_fn.to(device)

model_VGG.train()
train_step = 0
for epoch in range(50):
    for data in dataloader:
        images, labels,_ = data
        images,labels = images.to(device), labels.to(device)
        outputs = model_VGG(images)
        loss = loss_fn(outputs, labels)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        train_step += 1
        if train_step%10 == 0:
            print("epoch:{}, loss:{}".format(epoch, loss.item()))

torch.save(model_VGG.state_dict(),"./pth/pretrain_another.pth")

