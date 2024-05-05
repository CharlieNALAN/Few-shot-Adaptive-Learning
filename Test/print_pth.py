import torch
import torchvision

pth = torch.load("../pth/resnet50-11ad3fa6.pth")

print(pth)

model = torchvision.models.resnet50()
model.load_state_dict(pth)
model.fc = torch.nn.Linear(4096,10)
print(model)