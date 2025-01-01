import torch
from torchvision import transforms
from PIL import Image
from model import VGG16
import matplotlib.pyplot as plt
import os

# 设置中文字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']

# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

def test_model():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = VGG16().to(device)
    model.load_state_dict(torch.load("./models/VGG16__epoch10.pth", map_location=device))
    model.eval()

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28), interpolation=Image.LANCZOS),
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 读取 draws 文件夹中的图片
    image_folder = "../draws"
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if len(image_files) < 20:
        print("Warning: 图片数量不足 20 张，程序将处理所有图片！")

    # 初始化 Matplotlib 图表
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle("预测结果", fontsize=16)

    for idx, image_file in enumerate(image_files[:20]):  # 处理最多 20 张图片
        # 加载图片
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)

        # 转换为灰度图
        if image.mode != "L":
            image = image.convert("L")

        # 预处理图像
        transformed_image = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

        # 模型预测
        with torch.no_grad():
            output = model(transformed_image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_label = torch.max(probabilities, dim=1)

        # 输出预测结果
        print(f"图片: {image_file}, 预测数字: {predicted_label.item()}, 置信度: {confidence.item():.4f}")

        # 显示结果在 Matplotlib 图表上
        ax = axes[idx // 5, idx % 5]  # 计算子图位置
        ax.imshow(image, cmap="gray")
        ax.set_title(f"预测数字: {predicted_label.item()}\n置信度: {confidence.item():.4f}")
        ax.axis("off")

    # 移除多余的子图
    for i in range(len(image_files), 20):
        fig.delaxes(axes[i // 5, i % 5])

    # 显示图表
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 调整标题位置
    plt.show()


if __name__ == "__main__":
    test_model()
