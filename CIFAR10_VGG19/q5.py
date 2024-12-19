import os
import torch
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchsummary
import cv2
import numpy as np
from PyQt5 import QtWidgets


def showLabeledArgImg(button):
    path = QtWidgets.QFileDialog.getExistingDirectory(button, "Select Q5_1 Folder", os.getcwd()) + "/"
    imageFileNames = os.listdir(path)
    images = []
    plt.figure(figsize=(12, 12))
    for i, fileName in enumerate(imageFileNames):
        relPath = path + fileName
        image = Image.open(relPath)
        image.filename = relPath
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(30)
        ])
        image = transforms(image)
        images.append(image)
        plt.subplot(4, 3, i + 1)
        plt.imshow(image)
        plt.title(fileName.split(".")[0])
    plt.show()


def showModelStructure():
    model = torch.load("./models/vggModel.pth")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 選擇設備
    model = model.to(device)
    torchsummary.summary(model, (3, 32, 32))


def showModelAccLoss():
    lossPath = "./5_3_loss_1.png"
    accPath = "./5_3_acc_1.png"
    lossImage = cv2.imread(lossPath)
    accImage = cv2.imread(accPath)
    concatedImage = np.concatenate((lossImage, accImage), axis=0)
    cv2.imshow("Model Loss and Accuracy", concatedImage)


def inference(path, model, textLabel):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        predictResult = model(image)
    probs = torch.nn.functional.softmax(predictResult[0], dim=0).detach().cpu().numpy()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    _, predicted_idx = torch.max(predictResult, 1)
    predictedLabel = classes[predicted_idx.item()]
    textLabel.setText(f"Predicted= {predictedLabel}")
    plt.bar(classes, probs)
    plt.title("Probability of each class")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.show()
