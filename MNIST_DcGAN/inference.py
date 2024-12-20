import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from generator import Generator
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def run_inference(generator_path, dataset_path):
    """執行推論並顯示真實圖片和生成圖片"""
    # 初始化設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載訓練好的 Generator
    generator = Generator().to(device)
    try:
        generator.load_state_dict(torch.load(generator_path, map_location=device))
        generator.eval()
        print("Generator model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading the Generator model: {e}")
        return

    # 加載資料集
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = CustomDataset(root_dir=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 獲取一批真實圖片
    real_batch = next(iter(dataloader))

    # 生成假圖片
    noise = torch.randn(64, 100, 1, 1, device=device)
    fake_images = generator(noise).detach().cpu()

    # 繪製真實圖片與生成圖片
    plt.figure(figsize=(15, 15))

    # 真實圖片
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(make_grid(real_batch[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

    # 假圖片
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(make_grid(fake_images, padding=2, normalize=True), (1, 2, 0)))

    # 顯示圖片
    plt.show()
