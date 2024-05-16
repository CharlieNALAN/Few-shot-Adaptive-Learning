import torch
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet50_Weights, VGG16_Weights, ResNet18_Weights
from torchvision.transforms import transforms
import torch.nn.functional as F
import dataset

transform=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 以50%的概率水平翻转
        transforms.RandomRotation(30),  # 随机旋转±30度
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并调整为224x224
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机亮度和对比度
        # transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
dataset = dataset.TonguesDatasetInJson('../data',transform=transform)
total_size = len(dataset)
test_size = 40
train_size = total_size - test_size
#weights=ResNet18_Weights.IMAGENET1K_V1
model = torchvision.models.resnet18()
model.fc = nn.Linear(in_features=512,out_features=100)
model.add_module("fc2",nn.Linear(in_features=100,out_features=4))
# model = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# model.classifier[6] = nn.Linear(in_features=4096, out_features=4)
print(model)
loss_fn = nn.CrossEntropyLoss()
loss_fn2 = nn.KLDivLoss(reduction='batchmean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_fn.to(device)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=20)
test_loader = DataLoader(test_dataset, batch_size=10)
# dataset.detect()
# data=dataloader[0]
# print(data)
step = 0
for epoch in range(500):

    model.train()
    for data in train_loader:
        params = model.parameters()
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        log_probs = F.log_softmax(outputs, dim=1)  # 使用 log_softmax
        target_probs = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1.0)
        loss2 = loss_fn2(log_probs, target_probs)  # 使用正确的输入
        total_loss = 0.9*loss+ 0.1*loss2
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        step = step+1
        if step%10==0:
            print('Epoch {}, Step {}, Loss {}'.format(epoch,step,loss.item()))
    total_accuracy = 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            print(outputs.argmax(1))
            accuary = (outputs.argmax(1) == labels).sum()
            total_accuracy += accuary.item()
    print("整体测试集预测正确率：{}".format(total_accuracy / test_size))


# print(model)
# data,label = dataset[0]
# data = data.unsqueeze(0)
# output=model(data)
# print(data.shape)
# print(output)