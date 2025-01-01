import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # 第一层卷积层 (1, 28, 28) -> (64, 28, 28) -> 池化后 (64, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积层 (64, 14, 14) -> (192, 14, 14) -> 池化后 (192, 7, 7)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三层卷积层 (192, 7, 7) -> (384, 7, 7)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)

        # 第四层卷积层 (384, 7, 7) -> (256, 7, 7)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)

        # 第五层卷积层 (256, 7, 7) -> (256, 7, 7) -> 池化后 (256, 3, 3)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        # 输入特征为 (256 * 3 * 3 = 2304)
        self.fc1 = nn.Linear(in_features=256 * 3 * 3, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)  # 输出10个类别

        # Dropout层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一层卷积 + 激活 + 池化
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # 第二层卷积 + 激活 + 池化
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # 第三层卷积 + 激活
        x = F.relu(self.conv3(x))

        # 第四层卷积 + 激活
        x = F.relu(self.conv4(x))

        # 第五层卷积 + 激活 + 池化
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        # 展平操作
        x = x.view(-1, 256 * 3 * 3)

        # 全连接层1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 全连接层2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # 输出层
        x = self.fc3(x)

        return x

# 测试模型
# model = AlexNet()
# print(model)
