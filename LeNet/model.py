import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 第一层卷积层 (1, 28, 28) -> (6, 28, 28) -> 池化后 (6, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第二层卷积层 (6, 14, 14) -> (16, 10, 10) -> 池化后 (16, 5, 5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 全连接层
        # 输入特征为 (16 * 5 * 5 = 400)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)  # 输出10个类别

    def forward(self, x):
        # 第一层卷积 + 激活 + 池化
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # 第二层卷积 + 激活 + 池化
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # 展平操作
        x = x.view(-1, 16 * 5 * 5)

        # 全连接层1
        x = F.relu(self.fc1(x))

        # 全连接层2
        x = F.relu(self.fc2(x))

        # 输出层
        x = self.fc3(x)

        return x

# 测试模型
# model = LeNet()
# print(model)
