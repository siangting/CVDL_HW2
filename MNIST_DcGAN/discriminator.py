import torch
import torch.nn as nn
import torchsummary

# 定義 DCGAN 的 Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (3, 64, 64)
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 32, 32)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 16, 16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 8, 8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # (512, 4, 4)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: (1, 1, 1)
        )

    def forward(self, input):
        return self.main(input)


# 初始化模型權重
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 主程式
if __name__ == "__main__":
    # 確認設備 (使用 GPU 或 CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化 Discriminator
    netD = Discriminator().to(device)

    # 初始化模型權重
    netD.apply(weights_init)

    # 打印模型結構
    print(netD)

    # 使用 torchsummary 顯示模型結構摘要
    torchsummary.summary(netD, (3, 64, 64))  # Input 是 3x64x64 的圖像
