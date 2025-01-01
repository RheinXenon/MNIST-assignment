import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        """
        初始化数据集
        :param images_path: 图像文件路径
        :param labels_path: 标签文件路径
        :param transform: 数据预处理函数
        """
        self.images = self._load_images(images_path)
        self.labels = self._load_labels(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        返回索引 idx 的数据和标签
        :param idx: 索引
        :return: (image, label)
        """
        image = self.images[idx]
        label = self.labels[idx]

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_images(self, file_path):
        """
        加载图像数据
        :param file_path: 图像文件路径
        :return: numpy 数组 (num_images, 28, 28)
        """
        with open(file_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
        return images

    def _load_labels(self, file_path):
        """
        加载标签数据
        :param file_path: 标签文件路径
        :return: numpy 数组 (num_labels,)
        """
        with open(file_path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return labels


def get_dataloader(data_dir, batch_size=32, train_augment=True):
    """
    创建数据加载器
    :param data_dir: 数据集根目录
    :param batch_size: 批量大小
    :param train_augment: 是否对训练集进行数据增强
    :return: (train_loader, test_loader)
    """
    # 数据增强与预处理
    if train_augment:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),  # 将 NumPy 数组转换为 PIL 图像
            transforms.RandomRotation(15),  # 随机旋转，范围为 -15 到 15 度
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
            transforms.ToTensor(),  # 转换为 Tensor，并将像素值从 [0, 255] 缩放到 [0, 1]
            transforms.Normalize((0.5,), (0.5,))  # 归一化
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 文件路径
    train_images_path = os.path.join(data_dir, 'raw', 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'raw', 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(data_dir, 'raw', 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 'raw', 't10k-labels-idx1-ubyte')

    # 创建数据集
    train_dataset = MNISTDataset(train_images_path, train_labels_path, transform=transform_train)
    test_dataset = MNISTDataset(test_images_path, test_labels_path, transform=transform_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
