import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator


# 自定義 Dataset
class MNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # 確保圖像為 RGB 格式
        if self.transform:
            image = self.transform(image)
        return image


# 初始化模型權重
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 訓練設定
batch_size = 64
epochs = 10
latent_dim = 100
lr = 0.0002
beta1 = 0.5

# 資料加載與轉換
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 調整 MNIST 圖像大小
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNISTDataset(root_dir="../Q2_images/data/mnist", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)

# 定義損失函數和優化器
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 訓練過程
G_losses = []
D_losses = []

for epoch in range(epochs):
    for i, real_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
        real_data = real_data.to(device)
        batch_size = real_data.size(0)

        # 訓練 Discriminator
        netD.zero_grad()

        # 真實圖片損失
        label = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()

        # 假圖片損失
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_data = netG(noise)
        label.fill_(0.0)
        output = netD(fake_data.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()

        # 訓練 Generator
        netG.zero_grad()
        label.fill_(1.0)
        output = netD(fake_data).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        # 記錄損失
        D_losses.append(errD_real.item() + errD_fake.item())
        G_losses.append(errG.item())

# 保存模型
torch.save(netG.state_dict(), "./results/generator.pth")
torch.save(netD.state_dict(), "./results/discriminator.pth")

# 繪製損失曲線並保存
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses")
plt.savefig("./results/training_losses_40.jpg", format="jpg", dpi=300)
plt.show()
