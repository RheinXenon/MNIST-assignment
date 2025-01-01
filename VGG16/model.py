import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # 使用 torchvision 提供的 VGG16
        self.vgg16 = vgg16(pretrained=False)

        # 修改输入层：将输入通道数从 3 改为 1
        self.vgg16.features[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 修改分类层：将输出特征数改为 10（类别数）
        self.vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        # 使用修改后的 VGG16 前向传播
        return self.vgg16(x)