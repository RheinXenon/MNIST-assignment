import torch
from torchvision import transforms
from data_utils import get_dataloader
from PIL import Image
from model import SimpleCNN1
from train import train_model
import matplotlib.pyplot as plt


def train_mode():
    data_dir = "../datasets/MNIST"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = get_dataloader(data_dir, batch_size=128)
    model = SimpleCNN1().to(device)
    train_model(model, trainloader, testloader, device, num_epochs=10)


def test_mode():
    pass


def final_mode():
    image_path = "drawing.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN1().to(device)
    model.load_state_dict(torch.load("./models/CNN1_epoch10.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((28, 28)), 
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: 1 - x),  # 添加反色操作
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    image = Image.open(image_path)
    transformed_image = transform(image)  # 结果为 [channels, height, width]
    # plt.imshow(transformed_image[0].cpu(), cmap="gray")
    # plt.axis("off")
    # plt.show()

    transformed_image = transformed_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed_image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_label = torch.max(probabilities, dim=1)

    # 输出预测结果和置信度
    print(f"预测数字: {predicted_label.item()}, 置信度: {confidence.item():.4f}")
    return predicted_label.item(),confidence.item()



if __name__ == "__main__":
    final_mode()
