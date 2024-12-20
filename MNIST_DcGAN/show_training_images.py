import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

def show_training_images(data_root="../Q2_images/data/mnist/"):
    """顯示拼接後的原始與增強 MNIST 圖片"""
    # 確認資料夾存在
    if not os.path.exists(data_root):
        print(f"Error: Dataset folder '{data_root}' not found.")
        return

    # 收集所有圖片檔案
    all_images = [f for f in os.listdir(data_root) if f.endswith('.png')]
    if len(all_images) < 64:
        print(f"Error: Not enough images in '{data_root}'. At least 64 required.")
        return

    # 隨機選擇 64 張圖片
    random_images = random.sample(all_images, 64)

    # 定義圖片轉換
    transform_original = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    transform_augmented = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(60),
        transforms.ToTensor()
    ])

    # 載入圖片並進行轉換
    original_images = []
    augmented_images = []

    for img_name in random_images:
        img_path = os.path.join(data_root, img_name)
        img = Image.open(img_path).convert("L")  # 轉為灰度
        original_images.append(transform_original(img))
        augmented_images.append(transform_augmented(img))

    # 拼接原始圖片
    original_collage = torch.cat([torch.cat(original_images[i*8:(i+1)*8], dim=2) for i in range(8)], dim=1)
    augmented_collage = torch.cat([torch.cat(augmented_images[i*8:(i+1)*8], dim=2) for i in range(8)], dim=1)

    # 顯示拼接結果
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_collage.squeeze(0), cmap="gray")
    axes[0].set_title("Original Images")
    axes[0].axis("off")

    axes[1].imshow(augmented_collage.squeeze(0), cmap="gray")
    axes[1].set_title("Augmented Images")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
