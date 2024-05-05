import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, VGG16_Weights
import dataset
from dataset import TongueData, CustomSubset


# -----------------------------------------------------------------------------------------------------------
# 该方法已弃用，因为在subset子集执行add和remove方法时，相应封装已经对dataset基集进行了操作，无需再合并
def merge_subsets(subset1, subset2, root_dir):
    merged_dataset = TongueData(root_dir, empty=True)

    # 合并两个子集的信息
    all_image_paths = [subset1.dataset.image_paths[i] for i in subset1.indices] + \
                      [subset2.dataset.image_paths[i] for i in subset2.indices]
    all_image_labels = [subset1.dataset.image_labels[i] for i in subset1.indices] + \
                       [subset2.dataset.image_labels[i] for i in subset2.indices]

    # 去除重复的图像路径，因为同一图像可能在两个子集中都出现
    unique_image_paths = list(dict.fromkeys(all_image_paths))
    unique_image_labels = [all_image_labels[all_image_paths.index(path)] for path in unique_image_paths]

    # 更新 merged_dataset 的属性
    merged_dataset.image_paths = unique_image_paths
    merged_dataset.image_labels = unique_image_labels
    # 重新构建 class_to_idx 映射
    merged_dataset.class_to_idx = {label: idx for idx, label in enumerate(set(unique_image_labels))}

    return merged_dataset


# -----------------------------------------------------------------------------------------------------------


transform = transforms.Compose([
    transforms.Resize((192,144)),  # Todo 数据集具体大小具体分析，记得等真实数据集完成标注后修改
    transforms.ToTensor(),  # 转换为张量
])
model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
# train_dataset = dataset.TongueData("./dataset/train",transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64)
# test_dataset = dataset.TongueData("./dataset/val",transform=transform)
# test_loader = DataLoader(test_dataset,batch_size=64)
root_dir = './dataset/train'
dataset = TongueData(root_dir=root_dir, transform=transform)

# 计算训练集和测试集的大小
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% 作为训练集
test_size = total_size - train_size  # 剩下的 20% 作为测试集

# 分割数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
delete_dataset = TongueData(root_dir=root_dir, transform=transform, empty=True)
test_dataset = CustomSubset(dataset, test_dataset.indices)
train_dataset = CustomSubset(dataset, train_dataset.indices)
# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

Acc_th = 0.9
a = 0
learn_rate = 0.001
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


def seq(e):
    train_step = 0
    model.train()
    for epoch in range(e):
        for data in train_loader:
            imgs, labels, _ = data
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1
            if train_step % 20 == 0:
                print("Epoch:{}, Step:{}, Loss:{}".format(epoch, train_step, loss.item()))

    model.eval()
    # misclassified_indices = []
    misclassified_paths = []  # 用于存储预测错误的图像路径
    mislabel = []
    with torch.no_grad():
        for data in test_loader:
            imgs, labels, img_paths = data
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            # _, predicted = torch.max(outputs, 1)
            # misclassified = (predicted != labels).nonzero(as_tuple=False).squeeze()
            # misclassified_indices.extend(misclassified.tolist())
            _, predicted = torch.max(outputs, 1)
            misclassified = (predicted != labels).nonzero(as_tuple=False).squeeze()

            # 检查 misclassified 是否是零维张量，即标量 PS：如果不写这个，batch_size设置为1时会报错，直接把tensor干没了
            if misclassified.dim() == 0:
                if predicted[misclassified] != labels[misclassified]:
                    misclassified_paths.append(img_paths[misclassified])
                    mislabel.append(labels[misclassified])
            else:
                for idx in misclassified.tolist():
                    if predicted[idx] != labels[idx]:
                        misclassified_paths.append(img_paths[idx])
                        mislabel.append(labels[idx])

    # 删除预测错误的样本
    for idx in range(len(misclassified_paths)):
        test_dataset.remove(misclassified_paths[idx])  # test_dataset有一个可以通过图像路径删除样本的 remove 方法（已写好）
        delete_dataset.add(misclassified_paths[idx], mislabel[idx].item())


Acc = 0
epoch = 20
Threshold = 1
while Acc < Acc_th:
    seq(epoch)
    Acc = len(test_dataset) / (len(test_dataset) + len(
        delete_dataset) - a * Threshold)  # Todo:算法个人觉得能改进，保留意见，用a来扩充acc方法，如果删除图像太多，可能一轮就结束了，感觉应该和delete做系数
    a = a + len(delete_dataset)
    print("ACC:{}，a:{}".format(Acc, a))
    onew_dataset = dataset  # 这句其实不写也没事，但是方便阅读还是加上吧
    # 计算训练集和测试集的大小
    total_size = len(onew_dataset)
    train_size = int(0.8 * total_size)  # 80% 作为训练集
    test_size = total_size - train_size  # 剩下的 20% 作为测试集
    train_dataset, test_dataset = random_split(onew_dataset, [train_size, test_size])
    test_dataset = CustomSubset(onew_dataset, test_dataset.indices)
    train_dataset = CustomSubset(onew_dataset, train_dataset.indices)
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

print("----------------------------一段清洗完成-----------------------------------------")
print("清洗出{}张照片".format(len(delete_dataset)))
phase1_delete_num = len(delete_dataset)
print("-------------------------------------------------------------------------------")

model_VGG = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model_VGG.classifier[6] = nn.Linear(in_features=4096, out_features=2)
# print(model_VGG)
model_VGG.load_state_dict(torch.load("./pth/pretrain_another.pth"))
loss_fn2 = torch.nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(model_VGG.parameters())
model_VGG.to(device)
loss_fn2.to(device)
onew_loader = DataLoader(onew_dataset, batch_size=12, shuffle=True)

def train(e):
    train_step = 0
    model.train()
    model_VGG.train()
    for epoch_train in range(e):
        for data in onew_loader:
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)
            output1 = model(inputs)
            output2 = model_VGG(inputs)
            loss1 = loss_fn(output1, labels)
            loss2 = loss_fn2(output2, labels)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss1.backward()
            loss2.backward()
            optimizer.step()
            optimizer2.step()
            train_step += 1
            if train_step % 20 == 0:
                print("Epoch:{} , loss1={}, loss2={}".format(epoch_train, loss1.item(), loss2.item()))


def phase2_eval():
    # 在循环开始前定义一个列表来收集需要删除的路径
    # 必须！勿删已测试！因为dataloader不会实时更新，如果在验证过程中删除可能会导致dataloader的idx数组溢出
    paths_to_remove = []

    delete_dataloader = DataLoader(delete_dataset, batch_size=2, shuffle=True)
    model.eval()
    model_VGG.eval()
    with torch.no_grad():
        for data in delete_dataloader:
            inputs, labels, img_path = data
            inputs, labels = inputs.to(device), labels.to(device)
            output1 = model(inputs)
            output2 = model_VGG(inputs)

            _, preds1 = torch.max(output1, 1)
            _, preds2 = torch.max(output2, 1)

            # 逐个比较预测结果是否一致
            for i in range(inputs.size(0)):
                if preds1[i] == preds2[i]:
                    label = preds1[i].item()
                    path = img_path[i]
                    onew_dataset.add(path, label)
                    paths_to_remove.append(path)

        # 根据收集的路径列表删除项
        for path in paths_to_remove:
            delete_dataset.remove(path)


train(epoch)
phase2_eval()
phase2_delete_num = len(delete_dataset)
print("第二阶段重新加回{}张样本，清洗完成，现数据集有样本{}张，共清洗掉{}张".format(phase1_delete_num - phase2_delete_num,
                                                                                 len(dataset), phase2_delete_num))
print("--------------------------二阶段清洗完成------------------------------")
