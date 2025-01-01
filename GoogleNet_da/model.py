# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import googlenet

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()

        # 使用 torchvision 提供的 GoogLeNet，设置 aux_logits=False 禁用辅助分类器
        self.googlenet = googlenet(pretrained=False, aux_logits=False)

        # 修改输入层：将输入通道数从 3 改为 1
        self.googlenet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改第一个 maxpool 的 stride 和 kernel_size 适配更小的输入尺寸
        self.googlenet.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # 修改最终全连接层：将输出特征数改为 10（类别数）
        self.googlenet.fc = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        # 使用修改后的 GoogLeNet 前向传播
        return self.googlenet(x)

# 测试模型
if __name__ == "__main__":
    model = GoogleNet()
    print(model)

    # 创建一个 (1, 1, 28, 28) 的输入张量
    input_tensor = torch.randn(1, 1, 28, 28)
    output = model(input_tensor)
    print(output.shape)  # 输出应该是 (1, 10)
