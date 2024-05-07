import torch
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet50_Weights, VGG16_Weights, ResNet18_Weights
from torchvision.transforms import transforms

import dataset

transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
dataset = dataset.TonguesDatasetInJson('../data',transform=transform)
total_size = len(dataset)
test_size = 60
train_size = total_size - test_size
train_dataset, test_dataset = random_split(dataset, [train_size,test_size])
train_loader = DataLoader(train_dataset, batch_size=20)
test_loader = DataLoader(test_dataset, batch_size=10)
model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(in_features=512,out_features=100)
model.add_module("fc2",nn.Linear(in_features=100,out_features=4))
# model = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# model.classifier[6] = nn.Linear(in_features=4096, out_features=4)
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_fn.to(device)

# dataset.detect()
# data=dataloader[0]
# print(data)
step = 0
for epoch in range(500):
    model.train()
    for data in train_loader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
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
    print("整体测试集预测正确率：{}".format(total_accuracy / 60))


# print(model)
# data,label = dataset[0]
# data = data.unsqueeze(0)
# output=model(data)
# print(data.shape)
# print(output)