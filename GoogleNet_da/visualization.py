import struct
import numpy as np
import matplotlib.pyplot as plt

def load_images(file_path):
    with open(file_path, 'rb') as f:
        # 读取文件头部
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # 读取图像数据并转换为 numpy 数组
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        # 读取文件头部
        magic, num_labels = struct.unpack('>II', f.read(8))
        # 读取标签数据并转换为 numpy 数组
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# 加载训练图像和标签
images = load_images('../datasets/MNIST/raw/train-images-idx3-ubyte')
labels = load_labels('../datasets/MNIST/raw/train-labels-idx1-ubyte')

print(f"训练图像数量：{len(images)}, 每张图片尺寸：{images[0].shape}")

# 可视化48张图像和它们的标签
fig, axes = plt.subplots(6, 8, figsize=(10, 12))  # 创建一个6x8的子图网格
axes = axes.flatten()  # 将二维数组的子图展平

# 展示前48张图像，并显示对应的标签
for i in range(48):
    ax = axes[i]
    ax.imshow(images[i], cmap='gray')  # 显示图像，使用灰度色图
    ax.axis('off')  # 关闭坐标轴显示
    ax.set_title(f'Label: {labels[i]}', fontsize=10)  # 设置标题为标签

plt.tight_layout()  # 调整子图之间的间距
plt.show()
