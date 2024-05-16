import torch
from torch import nn, optim
import numpy as np

# 数据集生成器
def generate_data(num_tasks=10):
    dataset = []
    for _ in range(num_tasks):
        x = torch.from_numpy(np.random.rand(5, 1).astype(np.float32))  # 生成随机输入数据
        y = torch.from_numpy(np.random.rand(5, 1).astype(np.float32))  # 生成随机输出数据
        dataset.append((x, y))
    return dataset

# MAML算法
class MAML:
    def __init__(self, model):
        self.model = model
        self.meta_optimizer = optim.Adam(model.parameters(), lr=0.001)  # 元优化器，用于更新元模型的参数

    def inner_update(self, model, data, lr_inner=0.01, num_updates=1):
        optimizer = optim.SGD(model.parameters(), lr=lr_inner)  # 内部优化器，用于更新任务特定模型的参数
        x, y = data

        for _ in range(num_updates):
            loss = nn.MSELoss()(model(x), y)  # 计算损失
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

    def meta_update(self, tasks, iterations=100, num_updates=1):
        for _ in range(iterations):
            for task in tasks:
                data = task
                cloned_model = self.clone_model()  # 克隆模型
                self.inner_update(cloned_model, data, lr_inner=0.01, num_updates=num_updates)  # 内部更新

                for p, cloned_p in zip(self.model.parameters(), cloned_model.parameters()):

                    if p.grad is None:
                        p.grad = cloned_p.grad.clone()
                    else:
                        p.grad += cloned_p.grad.clone()  # 累加克隆模型的梯度
                self.meta_optimizer.step()  # 使用元优化器更新元模型的参数
#####好哥哥你为什么就是跑不通呢！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！####
    ####人都是会疯的￥@！#%￥……#@
    def clone_model(self):
        # 克隆模型的参数
        model_clone = SimpleModel()
        model_clone.load_state_dict(self.model.state_dict())  # 复制原始模型的参数到克隆模型
        return model_clone

# 简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)  # 简单的线性层

    def forward(self, x):
        return self.fc(x)  # 前向传播

# 生成数据集
tasks = generate_data(num_tasks=10)

# 初始化MAML和模型
model = SimpleModel()
maml = MAML(model)

# 进行元学习
maml.meta_update(tasks, iterations=100, num_updates=1)
