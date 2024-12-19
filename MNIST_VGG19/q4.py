import torch
import torchvision
import torchsummary
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io



def showModelStructure():
    path = "./models/vggModel.pth"
    model = torch.load(path)
    torchsummary.summary(model.cuda(), (3, 32, 32))


def showModelAccLoss(widget):
    imagePath = "./q4_loss_accuracy.png"
    widget.loadImage(imagePath)
    image = cv2.imread(imagePath)
    cv2.imshow("Model Loss and Accuracy", image)


def predict(model, png, label):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    image = Image.open(io.BytesIO(png.data())).convert("RGB")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        predictResult = model(image)

    probs = torch.nn.functional.softmax(predictResult[0], dim=0).detach().cpu().numpy()
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    _, predicted_idx = torch.max(predictResult, 1)
    predictedLabel = classes[predicted_idx.item()]
    label.setText(f"{predictedLabel}")
    plt.bar(classes, probs)
    plt.title("Probability of each class")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.show()


