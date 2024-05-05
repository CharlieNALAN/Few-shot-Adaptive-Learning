import torchvision
from torchvision.models import ResNet50_Weights

print(ResNet50_Weights.IMAGENET1K_V2)
model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
print(model)