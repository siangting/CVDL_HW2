import torch
from torchsummary import summary
from generator import Generator
from discriminator import Discriminator
import torch
import torch.nn as nn
import torchsummary

# 初始化模型權重
def weights_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def show_model_structure():
    """顯示 Generator 和 Discriminator 的模型結構"""
    # 確認設備
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 確認設備 (使用 GPU 或 CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化 Generator
    netG = Generator().to(device)
    # 初始化模型權重
    netG.apply(weights_init)
    # 打印模型結構
    print(netG)
    # 使用 torchsummary 顯示模型結構摘要
    torchsummary.summary(netG, (100, 1, 1))
    
    # 初始化 Discriminator
    netD = Discriminator().to(device)
    # 初始化模型權重
    netD.apply(weights_init)
    # 打印模型結構
    print(netD)
    # 使用 torchsummary 顯示模型結構摘要
    torchsummary.summary(netD, (3, 64, 64))  # Input 是 3x64x64 的圖像
