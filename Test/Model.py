import torchvision.models
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, VGG16_Weights
import torch
from torchvision.transforms import transforms

from dataset import TongueData

transform = transforms.Compose([
    transforms.Resize((768, 576)),
    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    # transforms.RandomRotation(15),  # 图像随机旋转±15度
    transforms.ToTensor(),  # 转换为张量
])
root_dir = "../Trimmed"
dataset = TongueData(root_dir=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
model_VGG = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model_VGG.classifier[6] = nn.Linear(in_features=4096, out_features=5)
# model_VGG=torchvision.models.resnet18(pretrained=True)
# model_VGG.fc = nn.Linear(in_features=512,out_features=5)
model_VGG.load_state_dict(torch.load("../pth/pretrain_VGG2.pth"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_VGG.to(device)
acc_num = 0
with torch.no_grad():
    for data in dataloader:
        images, labels,_ = data
        images, labels = images.to(device), labels.to(device)
        outputs = model_VGG(images)
        _, predicted = torch.max(outputs.data, 1)
        if predicted.item() == labels.item():
            acc_num += 1
        print(predicted.item(), "---",labels.item())
print(acc_num/len(dataset))