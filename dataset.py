import os

from PIL import Image
from torch.utils.data import Dataset, Subset


class TongueData(Dataset):
    def __init__(self, root_dir, transform=None, empty=False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.image_labels = []

        # 构建一个类别列表和索引
        self.class_to_idx = {}

        # 遍历根目录下的每个子目录
        for idx, directory in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, directory)
            if os.path.isdir(class_dir):
                self.class_to_idx[directory] = idx
                if empty is False:
                    # 遍历每个类别文件夹下的所有图像
                    for img_file in os.listdir(class_dir):
                        if img_file.endswith('.jpg') or img_file.endswith('.png') or img_file.endswith('.bmp'):
                            img_path = os.path.join(class_dir, img_file)
                            self.image_paths.append(img_path)
                            self.image_labels.append(directory)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.class_to_idx[self.image_labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

    def remove(self, path):
        # 检查路径是否存在于 image_paths 列表中
        if path in self.image_paths:
            # 获取路径的索引
            index = self.image_paths.index(path)
            # 从列表中移除对应的路径和标签
            del self.image_paths[index]
            del self.image_labels[index]
            print("文件:{}移除成功".format(path))
        else:
            print(f"Path {path} not found in dataset.")

    def add(self, path, label):
        # 检查提供的路径是否指向一个文件
        if not os.path.isfile(path):
            print(f"Path {path} is not a valid file.")
            return

        # 检查文件是否是图像
        if not (path.endswith('.jpg') or path.endswith('.png') or path.endswith('.bmp')):
            print(f"File {path} is not a valid image.")
            return

        idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # 使用反向映射找到对应的字符串标签
        if label in idx_to_class:
            label_str = idx_to_class[label]
        else:
            print(f"Label {label} does not exist in class_to_idx.")
            return

        # 添加图像路径到 image_paths
        self.image_paths.append(path)

        # 添加标签到 image_labels
        self.image_labels.append(label_str)

        print(f"文件: {path} 已添加到数据集中，标签为: {label_str}")


class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.indices = indices

    def remove(self, path):

        orig_index = None # 找到原始数据集中对应的索引
        for sub_idx, idx in enumerate(self.indices):
            if idx < len(self.dataset.image_paths) and self.dataset.image_paths[idx] == path:
                orig_index = idx
                self.indices.pop(sub_idx)  # 从 self.indices 中移除该索引
                break
        if orig_index is not None:
            self.dataset.remove(path)

            self.indices = [i-1 if i > orig_index else i for i in self.indices] # 更新 self.indices 中的所有后续索引
        else:
            print(f"Path {path} not found in subset.")




