import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# 定義資料夾路徑
data_root = "../Q2_images/data/mnist/"
all_images = [f for f in os.listdir(data_root) if f.endswith('.png')]

# 隨機選擇 64 張圖片
random_images = random.sample(all_images, 64)

# 定義原始圖片轉換和增強圖片轉換
transform_original = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
transform_augmented = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(60),
    transforms.ToTensor()
])

# 載入圖片
original_images = []
augmented_images = []

for img_name in random_images:
    img_path = os.path.join(data_root, img_name)
    img = Image.open(img_path).convert("L")  # 轉為灰度

    # 應用轉換
    original_images.append(transform_original(img))
    augmented_images.append(transform_augmented(img))

# 顯示原始圖片
fig1, axes1 = plt.subplots(8, 8, figsize=(8, 8))
fig1.suptitle("Original Images", fontsize=16)

for i in range(8):
    for j in range(8):
        idx = i * 8 + j
        ax = axes1[i, j]
        ax.imshow(original_images[idx].squeeze(0), cmap="gray")
        ax.axis("off")

# 顯示增強圖片
fig2, axes2 = plt.subplots(8, 8, figsize=(8, 8))
fig2.suptitle("Augmented Images", fontsize=16)

for i in range(8):
    for j in range(8):
        idx = i * 8 + j
        ax = axes2[i, j]
        ax.imshow(augmented_images[idx].squeeze(0), cmap="gray")
        ax.axis("off")

# 顯示兩個視窗
plt.show()
