import torch
import torchvision
import torchsummary

model = torchvision.models.vgg19_bn(num_classes=10)

torch.save(model, "./models/vggModel.pth")

torchsummary.summary(model.cuda(), (3, 32, 32))

# model = torch.load("./models/vggModel.pth")
# torchsummary.summary(model.cuda(), (3, 32, 32))